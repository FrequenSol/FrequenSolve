"""Internal HDF5 trace-store reader.

``TraceStore`` backs the public ``TraceDataset`` facade. ``RecordDatabase`` is
kept as a compatibility alias for older user code.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import h5py
import numpy as np
from xarray import DataArray

from frequensolve.seismic.trace_record import TraceRecord
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.simulation.sampling import UniformSweepSampling

__all__ = ["TraceStore", "RecordDatabase"]

try:
    import pyfftw

    pyfftw.interfaces.cache.enable()
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.config.NUM_THREADS = 4
except ImportError:
    warnings.warn("pyfftw not found, using numpy for FFT (slow)")
    import numpy.fft as fft


def process_string(raw):
    if isinstance(raw, (bytes, bytearray)):
        s = raw.decode("utf-8", "ignore").rstrip()
    else:
        s = raw.tobytes().decode("utf-8", "ignore").rstrip()
    return s


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

    def __init__(
        self,
        metadata: Dict[str, Any],
        files: Optional[Iterable[Union[str, Path]]] = None,
        *,
        records: Optional[Iterable[Union[str, Path]]] = None,
        upscale: int = 1,
    ):
        record_files = _as_file_list(records) if records is not None else None
        if files is None:
            files = record_files
        else:
            files = _as_file_list(files)
        if record_files is not None and files != record_files:
            raise ValueError("files and legacy records arguments do not match")
        if files is None:
            raise TypeError("TraceStore requires trace files")
        self.metadata = metadata
        self.files = [str(file) for file in files]
        self.upscale = upscale
        self._consolidated = None

    @property
    def records(self) -> List[str]:
        """Compatibility alias for ``files``."""
        return self.files

    @records.setter
    def records(self, value: Iterable[Union[str, Path]]) -> None:
        self.files = [str(file) for file in value]

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
        db.consolidate_h5()
        return db

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
            return [
                name
                for name, item in f.items()
                if isinstance(item, h5py.Dataset)
                and name not in {"frequency", "laplace", "source_file"}
            ]

    def dims(self, group) -> list[str]:
        self._ensure_consolidated()
        with h5py.File(self._consolidated, "r") as f:
            return _decode_dim_list(f[group].attrs["dims"])

    def components(self, group) -> list[str]:
        self._ensure_consolidated()
        with h5py.File(self._consolidated, "r") as f:
            path = f"survey/receiver_groups/{group}/traces/component_name"
            if path in f:
                return _unique_preserve_order(_decode_h5_strings(f[path][()]))
            if "component" in f[group].attrs:
                return _decode_h5_strings(f[group].attrs["component"])
            return np.arange(1, f[group].shape[-2] + 1)

    def shots(self, group) -> list[str]:
        self._ensure_consolidated()
        with h5py.File(self._consolidated, "r") as f:
            path = f"survey/receiver_groups/{group}/traces/source_id"
            if path in f:
                return _unique_preserve_order(f[path][()])
            if "shot" in f[group].attrs:
                return f[group].attrs["shot"]
            return np.arange(1, f[group].shape[-1] + 1)

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
            path = f"survey/receiver_groups/{group}/traces/receiver_id"
            if path in f:
                return _unique_preserve_order(f[path][()])
            if "receiver" in f[group].attrs:
                return f[group].attrs["receiver"]
            dims = np.asarray(_decode_dim_list(f[group].attrs["dims"])[::-1])
            ind = np.where(dims == "receiver")[0][0]
            return np.arange(1, f[group].shape[ind] + 1)

    def survey_tables(self) -> Dict[str, Any]:
        if self._consolidated is None:
            self.consolidate_h5()

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
        if self._consolidated is None:
            self.consolidate_h5()

    @staticmethod
    def _consolidated_path(first_record: Union[str, Path]) -> Path:
        first_record = Path(first_record)
        stem = first_record.stem
        prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
        return first_record.with_name(f"{prefix}_consolidated.h5")

    @staticmethod
    def _read_trace_frequency(file: Path) -> float:
        with h5py.File(file, "r") as h5:
            if "frequency" not in h5:
                raise KeyError(f"'frequency' dataset not found in {file}")
            return float(h5["frequency"][()])

    def consolidate_h5(self):
        """
        Creates a virtual HDF5 dataset that combines datasets from each frequency file
        along a new dimension. This allows efficient access to the full dataset without
        loading all data into memory.
        """

        valid_records = []
        freqs = []
        for record in [Path(file) for file in self.files]:
            if not record.exists():
                warnings.warn(
                    f"Trace file does not exist and will be skipped: {record}"
                )
                continue
            try:
                freqs.append(self._read_trace_frequency(record))
                valid_records.append(record)
            except Exception as exc:
                warnings.warn(
                    f"Trace file could not be read and will be skipped: {record}: {exc}"
                )

        if not valid_records:
            raise FileNotFoundError("No readable trace files were found")

        new_file = self._consolidated_path(valid_records[0])
        if new_file.exists():
            new_file.unlink()

        with h5py.File(new_file, "w") as nf:
            nf.create_dataset("frequency", data=np.asarray(freqs, dtype=float))
            string_dtype = h5py.string_dtype(encoding="utf-8")
            nf.create_dataset(
                "source_file",
                data=np.asarray(
                    [str(file) for file in valid_records], dtype=string_dtype
                ),
            )

            for file in valid_records:
                with h5py.File(file, "r") as f:
                    if "survey" in f:
                        f.copy("survey", nf)
                        break

            for group in self.metadata["groups"]:
                with h5py.File(valid_records[0], "r") as f:
                    if group not in f:
                        raise KeyError(
                            f"Group '{group}' not found in HDF5 {valid_records[0]}"
                        )
                    dset_shape = f[group].shape
                    dset_dtype = f[group].dtype
                    dims = _decode_dim_list(f[group].attrs["dims"])
                    coords = {}
                    for dim in dims:
                        if dim in f[group].attrs:
                            coord = f[group].attrs[dim]
                        else:
                            coord = np.arange(1, dset_shape[dims[::-1].index(dim)] + 1)
                        coords[dim] = coord

                # Create virtual layout for data
                layout = h5py.VirtualLayout(
                    shape=(len(valid_records),) + dset_shape, dtype=dset_dtype
                )
                for i, file in enumerate(valid_records):
                    vsource = h5py.VirtualSource(str(file), group, shape=dset_shape)
                    layout[i] = vsource
                nf.create_virtual_dataset(group, layout)

                dims.append("frequency")
                coords["frequency"] = freqs

                dset = nf[group]
                dset.attrs["dims"] = dims
                for dim, coord in coords.items():
                    if len(coord) < 10000:
                        dset.attrs[dim] = coord
        self._consolidated = Path(new_file)

    def read_h5(self, group: str) -> TraceRecord:
        import dask.array as da

        self._ensure_consolidated()
        f = h5py.File(self._consolidated, "r")
        dset = f[group]
        dims = _decode_dim_list(dset.attrs["dims"])
        coords = {}
        for dim in dims:
            if dim in dset.attrs:
                coords[dim] = dset.attrs[dim]
            else:
                idx = np.where(np.array(dims[::-1]) == dim)[0][0]
                coords[dim] = np.arange(1, dset.shape[idx] + 1)
        tmp = ["complex"]
        for dim in dims:
            tmp.append(dim)
        dims = tmp
        coords["complex"] = ["real", "imag"]

        chunks = (dset.shape[0], 1, 1, *dset.shape[3:])
        data = da.from_array(dset, chunks=chunks)
        fd = TraceRecord(data, dims=dims[::-1], coords=coords)
        return fd

    def read_FD(
        self,
        group: str,
        component: str,
        shot: int,
        wavelet: Optional[Wavelet] = None,
        **kwargs,
    ):
        if wavelet is None:
            dset = self.read_h5(group)
            gather = dset.sel(component=component, shot=shot)
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
            gather = dset.sel(component=component, shot=shot)

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
        shot: int,
        wavelet: Wavelet,
        upscale: int = 1,
        T_max: Optional[float] = None,
        **kwargs,
    ) -> TraceRecord:
        sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=self.metadata["f_max"],
            df=self.metadata["df"],
            upscale=upscale,
        )
        fd = self.read_FD(group, component, shot, wavelet, **kwargs)
        wavelet.times = sampling.T_list

        fd = fd.interp(frequency=sampling.F_list, kwargs={"fill_value": 0})
        td = fft.irfft(fd.data, axis=0)
        dims = ["time" if d == "frequency" else d for d in fd.dims]
        coords = {}
        for d in dims:
            if d in fd.coords:
                coords[d] = fd.coords[d]
            else:
                coords[d] = sampling.T_list[:-1] - wavelet.center

        td = TraceRecord(data=td, dims=dims, coords=coords)
        if T_max is not None:
            td = td.sel(time=slice(None, T_max))

        td.attrs["source_group"] = shot
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
        shot: int,
        wavelet: Wavelet,
        window: DataArray,
        N_window: int,
        upscale: int = 1,
        T_max: Optional[float] = None,
        **kwargs,
    ) -> TraceRecord:
        sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=self.metadata["f_max"],
            df=self.metadata["df"],
            upscale=upscale,
        )
        fd = self.read_FD(group, component, shot, wavelet, **kwargs)
        wavelet.times = sampling.T_list

        fd = fd.interp(frequency=sampling.F_list, kwargs={"fill_value": 0})

        N = len(window.data)

        W = np.fft.fft(window.data) / N
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

        td = TraceRecord(data=td, dims=dims, coords=coords)
        if T_max is not None:
            td = td.sel(time=slice(None, T_max))

        td.attrs["source_group"] = shot
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


RecordDatabase = TraceStore
