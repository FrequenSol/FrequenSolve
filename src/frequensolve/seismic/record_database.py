"""
Not used yet; this will complement survey.py for reading and visualizing
data when finished.

Right now this is just a hodge-podge of code that was displaced in
the refactoring process.
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
from xarray import DataArray

from frequensolve.seismic.shot_record import ShotRecord
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.simulation.sampling import UniformSweepSampling

try:
    import pyfftw

    pyfftw.interfaces.cache.enable()
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.config.NUM_THREADS = 4
except:
    warnings.warn("pyfftw not found, using numpy for FFT (slow)")
    import numpy.fft as fft


def process_string(raw):
    if isinstance(raw, (bytes, bytearray)):
        s = raw.decode("utf-8", "ignore").rstrip()
    else:
        s = raw.tobytes().decode("utf-8", "ignore").rstrip()
    return s


@dataclass
class RecordDatabase:
    metadata: Dict[str, Any]
    records: List[str]
    _upscale: int
    _consolidated: Optional[Path] = None

    def __init__(self, metadata: Dict[str, Any], records: List[str], upscale: int = 1):
        self.metadata = metadata
        self.records = records
        self.upscale = upscale

    @classmethod
    def from_job(cls, job, upscale: int = 1):
        """Create a RecordDatabase from a dictionary of results.

        Args:
            results: A dictionary of results from a Frontera job.
            proj_path: The path to the project directory.

        Returns:
            A RecordDatabase object.
        """
        records = job.records
        proj_path = Path(job.project_path).resolve()

        f_map = records["frequencies"]
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
            "simulation": proj_path / records["simulation"],
            "groups": records["groups"],
            "df": df,
            "f_max": f_max,
            "f_map": f_map,
        }

        files = records["files"]
        db = cls(metadata=meta, records=files, upscale=upscale)
        db.consolidate_h5()
        return db

    @property
    def upscale(self) -> int:
        return self._upscale

    @upscale.setter
    def upscale(self, upscale: int) -> None:
        self._upscale = upscale

    def times(self, upscale: Optional[int] = None) -> np.ndarray:
        """Returns the times of the records."""

        upscale = self.upscale if upscale is None else upscale
        sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=self.metadata["f_max"],
            df=self.metadata["df"],
            upscale=upscale,
        )
        return sampling.T_list

    def __len__(self) -> int:
        """Returns the number of records in the database."""
        size = 0
        for group in self.groups:
            recv = self.receivers(group)
            shot = self.shots(group)
            comp = self.components(group)
            size += len(recv) * len(shot) * len(comp)
        return size

    @property
    def groups(self) -> list[str]:
        f = h5py.File(self._consolidated, "r")
        return list(f.keys())

    def dims(self, group) -> list[str]:
        f = h5py.File(self._consolidated, "r")
        dims = f[group].attrs["dims"]
        return dims

    def components(self, group) -> list[str]:
        f = h5py.File(self._consolidated, "r")
        return f[group].attrs["component"]

    def shots(self, group) -> list[str]:
        f = h5py.File(self._consolidated, "r")
        return f[group].attrs["shot"]

    def frequencies(self, group) -> list[str]:
        f = h5py.File(self._consolidated, "r")
        return f[group].attrs["frequency"]

    def receivers(self, group) -> list[str]:
        f = h5py.File(self._consolidated, "r")

        if "receiver" in f[group].attrs:
            recv = f[group].attrs["receiver"]
        else:
            recv = np.arange(1, f[group].shape[0] + 1)
        return recv

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

    def consolidate_h5(self):
        """Consolidate records into single h5 files to improve efficiency and convenience.

        Creates a virtual HDF5 dataset that combines datasets from each frequency file
        along a new dimension. This allows efficient access to the full dataset without
        loading all data into memory.
        """

        file = self.records[0]
        base = "_".join(file.split("_")[:-1])
        new_file = f"{base}_consolidated.h5"
        if os.path.exists(new_file):
            os.remove(new_file)
        freqs = []

        with h5py.File(new_file, "w") as nf:
            for i, file in enumerate(self.records):
                with h5py.File(file, "r") as f:
                    freq = f["frequency"][()]
                freqs.append(freq)

            for group in self.metadata["groups"]:

                # Get shape and dimensions
                with h5py.File(self.records[0], "r") as f:
                    if group not in f:
                        raise KeyError(
                            f"Group '{group}' not found in HDF5 {self.records[0]}"
                        )
                    dset_shape = f[group].shape
                    dset_dtype = f[group].dtype
                    dims = f[group].attrs["dims"]
                    dims = [process_string(d) for d in dims]
                    coords = {}
                    for dim in dims:
                        if dim in f[group].attrs:
                            coord = f[group].attrs[dim]
                            if isinstance(coord[0], bytes):
                                coord = [process_string(c) for c in coord]
                            else:
                                coord = np.array(coord).tolist()
                        else:
                            coord = np.arange(1, dset_shape[dims[::-1].index(dim)] + 1)
                        coords[dim] = coord

                # Create virtual layout for data
                layout = h5py.VirtualLayout(
                    shape=(len(self.records),) + dset_shape, dtype=dset_dtype
                )
                for i, file in enumerate(self.records):
                    vsource = h5py.VirtualSource(file, group, shape=dset_shape)
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

    def read_h5(self, group: str) -> ShotRecord:
        import dask.array as da

        f = h5py.File(self._consolidated, "r")
        dset = f[group]
        dims = dset.attrs["dims"]
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
        fd = ShotRecord(data, dims=dims[::-1], coords=coords)
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
    ) -> ShotRecord:
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

        td = ShotRecord(data=td, dims=dims, coords=coords)
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
    ) -> ShotRecord:
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

        td = ShotRecord(data=td, dims=dims, coords=coords)
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
