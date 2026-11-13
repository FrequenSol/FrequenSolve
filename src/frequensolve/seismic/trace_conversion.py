"""Bounded conversion of arrays and SEG-Y traces to Sauce observed-data stores."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import h5py
import numpy as np
import xarray as xr

from frequensolve._optional import optional_dependency_error

__all__ = ["convert_traces"]


def _ids(values: Any, name: str) -> np.ndarray:
    values = np.asarray(values)
    if (
        values.dtype.kind not in "iu"
        or np.any(values < 1)
        or np.any(values > 2**31 - 1)
    ):
        raise ValueError(f"{name} must contain positive int32 identifiers")
    return values.astype(np.int32)


def _axis(values: Any, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a nonempty finite numeric axis")
    return values


def _sampling(time: Any, dt: float | None, t0: float | None) -> tuple[float, float]:
    time = _axis(time, "time") if time is not None else None
    if time is not None:
        if len(time) > 1:
            spacing = float(time[1] - time[0])
            if not np.allclose(np.diff(time), spacing, rtol=1e-7, atol=1e-12):
                raise ValueError(
                    "Sauce time stores require uniformly sampled time coordinates"
                )
            if dt is not None and not np.isclose(dt, spacing, rtol=1e-7, atol=1e-12):
                raise ValueError("dt conflicts with time coordinates")
            dt = spacing
        if t0 is not None and not np.isclose(t0, time[0], rtol=1e-7, atol=1e-12):
            raise ValueError("t0 conflicts with time coordinates")
        t0 = float(time[0])
    if dt is None or not np.isfinite(dt) or dt <= 0:
        raise ValueError("A positive dt in seconds is required")
    t0 = 0.0 if t0 is None else float(t0)
    if not np.isfinite(t0):
        raise ValueError("t0 must be finite")
    return float(dt), t0


class _ArrayTraces:
    def __init__(
        self,
        data: Any,
        *,
        dims: Sequence[str] | None,
        variable: str | None,
        dt: float | None,
        t0: float | None,
        source_ids: Any,
        receiver_ids: Any,
        component: str | None,
    ) -> None:
        numpy_input = data if isinstance(data, np.ndarray) else None
        self.npy_file: Path | None = None
        self.npy_offset = 0
        self.matrix: np.ndarray | None = None
        self.sample_major = False
        if isinstance(data, xr.Dataset):
            if variable is None:
                if len(data.data_vars) != 1:
                    raise ValueError(
                        "Select variable when converting a multi-variable Dataset"
                    )
                data = data[next(iter(data.data_vars))]
            else:
                data = data[variable]
        if not isinstance(data, xr.DataArray):
            if dims is None:
                if data.ndim != 2:
                    raise ValueError("Provide dims for arrays other than (trace, time)")
                dims = ("trace", "time")
            data = xr.DataArray(data, dims=dims)
        elif dims is not None:
            raise ValueError("dims is only used for NumPy inputs")
        domains = [name for name in ("time", "frequency") if name in data.dims]
        if len(domains) != 1:
            raise ValueError("Traces require exactly one time or frequency dimension")
        self.domain = domains[0]
        self.data = data
        self.trace_dims = [d for d in data.dims if d not in (self.domain, "complex")]
        if not self.trace_dims or any(
            d not in {"trace", "receiver", "source", "component"}
            for d in self.trace_dims
        ):
            raise ValueError(
                "Trace dimensions must be source, receiver, component, or trace"
            )
        self.shape = tuple(data.sizes[d] for d in self.trace_dims)
        self.ntrace = int(np.prod(self.shape))
        self.nsample = data.sizes[self.domain]
        if not self.ntrace or not self.nsample or self.ntrace > 2**31 - 1:
            raise ValueError(
                "Store requires nonempty data and at most int32 trace count"
            )
        self.component = component
        self.source_ids = None if source_ids is None else np.asarray(source_ids)
        self.receiver_ids = None if receiver_ids is None else np.asarray(receiver_ids)
        self.dt: float | None = None
        self.t0: float | None = None
        self.frequency: np.ndarray | None = None
        self.laplace: np.ndarray | None = None
        self.time_power: np.ndarray | None = None
        if self.domain == "time":
            if data.dtype.kind not in "fiu" or "complex" in data.dims:
                raise ValueError("Time samples must be real numeric values")
            self.dt, self.t0 = _sampling(
                data.coords["time"] if "time" in data.coords else None, dt, t0
            )
        else:
            if dt is not None or t0 is not None:
                raise ValueError("dt and t0 only apply to time traces")
            if "frequency" not in data.coords:
                raise ValueError("Frequency traces require frequency coordinates in Hz")
            self.frequency = _axis(data.frequency, "frequency")
            self.laplace = self._spectral_axis("laplace", 0.0)
            self.time_power = self._spectral_axis("time_power", 0)
            if np.any(self.frequency < 0) or np.any(self.laplace > 0):
                raise ValueError(
                    "frequency must be nonnegative and laplace nonpositive"
                )
            if np.any(self.time_power < 0) or np.any(
                self.time_power != np.floor(self.time_power)
            ):
                raise ValueError("time_power must contain nonnegative integers")
            if np.any(self.time_power > 2**31 - 1):
                raise ValueError("time_power exceeds int32")
            coordinates = np.column_stack(
                (self.frequency, self.laplace, self.time_power)
            )
            if len(np.unique(coordinates, axis=0)) != len(coordinates):
                raise ValueError("Duplicate frequency/laplace/time_power coordinates")
            if "complex" in data.dims:
                if list(data.coords.get("complex", [])) != ["real", "imag"]:
                    raise ValueError(
                        "Split complex coordinates must be ['real', 'imag']"
                    )
            elif data.dtype.kind not in "fciu":
                raise ValueError("Frequency samples must be numeric")
        if numpy_input is not None:
            canonical = np.moveaxis(numpy_input, data.get_axis_num(self.domain), -1)
            if canonical.ndim == 2 or canonical.flags.c_contiguous:
                self.matrix = canonical.reshape(self.ntrace, self.nsample)
            self.sample_major = (
                numpy_input.flags.c_contiguous
                and data.dims[0] == self.domain
                and self.domain == "time"
            )
        if self.sample_major:
            assert numpy_input is not None
            self.matrix = numpy_input.reshape(self.nsample, self.ntrace).T
        elif (
            self.matrix is not None
            and self.matrix.flags.f_contiguous
            and self.domain == "time"
        ):
            self.sample_major = True
        self.provenance: dict[str, Any] = {
            "format": "xarray",
            "dimensions": list(data.dims),
        }
        acquisition = {}
        for key in (
            "units",
            "source_strength",
            "source_strength_units",
            "source_signature_hash",
            "source_signature_applied",
            "receiver_response_applied",
            "wavelet_definition",
            "wavelet",
            "wavelet_center",
        ):
            if key in data.attrs:
                value = data.attrs[key]
                acquisition[key] = (
                    value.item() if isinstance(value, np.generic) else value
                )
        if acquisition:
            self.provenance["acquisition"] = acquisition

    def _spectral_axis(self, name: str, default: float) -> np.ndarray:
        if name not in self.data.coords:
            return np.full(self.nsample, default)
        coord = self.data.coords[name]
        if coord.dims not in [(), ("frequency",)]:
            raise ValueError(f"{name} must be scalar or indexed by frequency")
        return _axis(np.broadcast_to(coord.values, (self.nsample,)), name)

    def _column(
        self, start: int, stop: int, name: str, alias: str, override: Any, default: Any
    ) -> np.ndarray:
        indexes = np.unravel_index(np.arange(start, stop), self.shape)
        if override is not None:
            values = np.asarray(override)
            if values.ndim == 0:
                return np.full(stop - start, values.item())
            if values.shape != (self.ntrace,):
                raise ValueError(
                    f"{name} must be scalar or have one entry per flattened trace"
                )
            return values[start:stop]
        coord = (
            self.data.coords[name]
            if name in self.data.coords
            else self.data.coords[alias] if alias in self.data.coords else None
        )
        if coord is not None:
            if any(d not in self.trace_dims for d in coord.dims):
                raise ValueError(f"{name} must not vary over samples")
            selected = coord.isel(
                {
                    d: xr.DataArray(indexes[self.trace_dims.index(d)], dims="_row")
                    for d in coord.dims
                }
            )
            return np.broadcast_to(selected.values, (stop - start,))
        if alias in self.trace_dims:
            return indexes[self.trace_dims.index(alias)] + 1
        return (
            np.full(stop - start, default)
            if default is not None
            else np.arange(start + 1, stop + 1)
        )

    def component_names(self) -> Sequence[str]:
        values: Any
        if self.component is not None:
            values = self.component
        elif "component" in self.data.coords:
            values = self.data.coords["component"].values
        elif "component" in self.data.dims:
            values = np.arange(1, self.data.sizes["component"] + 1)
        else:
            values = self.data.attrs.get("long_name", self.data.name or "")
        names = np.unique(np.asarray(values).astype(str))
        if np.any(names == ""):
            raise ValueError(
                "Specify component or provide component coordinates/long_name"
            )
        return names

    def metadata(self, start: int, stop: int) -> dict[str, np.ndarray]:
        source = self._column(
            start,
            stop,
            "source_id",
            "source",
            self.source_ids,
            self.data.attrs.get("source_id", 1),
        )
        receiver = self._column(
            start, stop, "receiver_id", "receiver", self.receiver_ids, None
        )
        component = self._column(
            start,
            stop,
            "component",
            "component",
            self.component,
            self.data.attrs.get("long_name", self.data.name or ""),
        )
        if np.any(np.asarray(component).astype(str) == ""):
            raise ValueError(
                "Specify component or provide component coordinates/long_name"
            )
        return {
            "source_id": _ids(source, "source_id"),
            "receiver_id": _ids(receiver, "receiver_id"),
            "component_name": component,
        }

    def _read_npy_window(
        self, offset: int, shape: tuple[int, int], transpose: bool = False
    ) -> np.ndarray:
        # Read directly into a bounded NumPy buffer. Unlike a file-wide mmap,
        # processed pages do not remain part of this process's resident set.
        assert self.npy_file is not None
        count = int(np.prod(shape))
        values = np.fromfile(
            self.npy_file,
            dtype=self.data.dtype,
            count=count,
            offset=self.npy_offset + offset * self.data.dtype.itemsize,
        )
        if values.size != count:
            raise ValueError("NPY input was truncated during conversion")
        values = values.reshape(shape)
        return values.T if transpose else values

    def read_sample_tile(self, first: int, last: int) -> np.ndarray:
        if self.npy_file is not None:
            return self._read_npy_window(
                first * self.ntrace, (last - first, self.ntrace), transpose=True
            )
        assert self.matrix is not None
        return self.matrix[:, first:last]

    def read(self, start: int, stop: int) -> np.ndarray:
        if self.npy_file is not None:
            if self.matrix is not None and self.matrix.flags.c_contiguous:
                return self._read_npy_window(
                    start * self.nsample, (stop - start, self.nsample)
                )
            mapping = np.load(self.npy_file, mmap_mode="r", allow_pickle=False)
            try:
                canonical = np.moveaxis(
                    mapping, self.data.get_axis_num(self.domain), -1
                )
                indexes = np.unravel_index(np.arange(start, stop), self.shape)
                return np.array(canonical[indexes], copy=True)
            finally:
                mapping._mmap.close()
        if self.matrix is not None:
            return self.matrix[start:stop]
        if len(self.trace_dims) == 1:
            dim = self.trace_dims[0]
            selected = self.data.isel({dim: slice(start, stop)}).rename({dim: "_row"})
            return self._values(selected)
        indexes = np.unravel_index(np.arange(start, stop), self.shape)
        selected = self.data.isel(
            {
                d: xr.DataArray(indexes[i], dims="_row")
                for i, d in enumerate(self.trace_dims)
            }
        )
        return self._values(selected)

    def _values(self, selected: xr.DataArray) -> np.ndarray:
        if "complex" in selected.dims:
            selected = selected.sel(complex="real") + 1j * selected.sel(complex="imag")
        return np.asarray(selected.transpose("_row", self.domain).values)


class _SegyTraces:
    def __init__(
        self,
        file: str | Path,
        *,
        component: str | None,
        headers: Mapping[str, int] | None,
        endian: str,
        source_ids: Any,
        receiver_ids: Any,
    ) -> None:
        try:
            import segyio
        except ImportError as error:
            raise optional_dependency_error(
                "SEG-Y trace conversion",
                extra="seismic-io",
                error=error,
                dependencies=("segyio",),
            ) from None
        self.domain = "time"
        self.source_ids = None if source_ids is None else np.asarray(source_ids)
        self.receiver_ids = None if receiver_ids is None else np.asarray(receiver_ids)
        self.file = segyio.open(str(file), "r", ignore_geometry=True, endian=endian)
        try:
            self.ntrace = self.file.tracecount
            self.nsample = len(self.file.samples)
            if not self.ntrace or not self.nsample or self.ntrace > 2**31 - 1:
                raise ValueError(
                    "SEG-Y requires nonempty data and an int32 trace count"
                )
            if component is None or not isinstance(component, str) or not component:
                raise ValueError("SEG-Y requires an explicit component name")
            self.component = component
            fields = segyio.TraceField
            self.headers = {
                "source_id": int(fields.FieldRecord),
                "receiver_id": int(fields.TraceNumber),
                "dt": int(fields.TRACE_SAMPLE_INTERVAL),
                "t0": int(fields.DelayRecordingTime),
            }
            if headers:
                if set(headers) - self.headers.keys():
                    raise ValueError(
                        "segy_headers keys must be source_id, receiver_id, dt, t0"
                    )
                self.headers.update(headers)
            self.binary_dt = self.file.bin[segyio.BinField.Interval]
            self.dt: float | None = None
            self.t0: float | None = None
            self.frequency: np.ndarray | None = None
            self.laplace: np.ndarray | None = None
            self.time_power: np.ndarray | None = None
            self.provenance = {
                "format": "segy",
                "path": str(Path(file).resolve()),
                "headers": self.headers,
                "endian": endian,
            }
            self.metadata(0, min(self.ntrace, 1))
        except BaseException:
            self.file.close()
            raise

    def component_names(self) -> Sequence[str]:
        return [self.component]

    def metadata(self, start: int, stop: int) -> dict[str, np.ndarray]:
        cols = {
            name: np.asarray(self.file.attributes(field)[start:stop])
            for name, field in self.headers.items()
        }
        dt = np.where(cols["dt"] == 0, self.binary_dt, cols["dt"]) * 1e-6
        t0 = cols["t0"] * 1e-3
        if self.dt is None:
            self.dt, self.t0 = _sampling(None, float(dt[0]), float(t0[0]))
        assert self.dt is not None and self.t0 is not None
        if not np.allclose(dt, self.dt, rtol=1e-7, atol=1e-12) or not np.allclose(
            t0, self.t0, rtol=1e-7, atol=1e-12
        ):
            raise ValueError(
                "SEG-Y traces have differing dt/t0; align sampling before conversion"
            )
        for name in ("source_id", "receiver_id"):
            override = getattr(self, name.replace("_id", "_ids"))
            if override is not None:
                override = np.asarray(override)
                if override.ndim == 0:
                    cols[name] = np.full(stop - start, override.item())
                elif override.shape == (self.ntrace,):
                    cols[name] = override[start:stop]
                else:
                    raise ValueError(
                        f"{name} override must be scalar or have one entry per trace"
                    )
        return {
            "source_id": _ids(cols["source_id"], "source_id"),
            "receiver_id": _ids(cols["receiver_id"], "receiver_id"),
            "component_name": np.full(stop - start, self.component),
        }

    def read(self, start: int, stop: int) -> np.ndarray:
        # The C reader fills the entire bounded block, without a Python trace loop.
        return self.file.trace.raw[start:stop]


def _checked_samples(raw: np.ndarray, domain: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        samples = np.asarray(
            raw, dtype=np.float32 if domain == "time" else np.complex64
        )
    if not np.all(np.isfinite(samples)):
        if not np.all(np.isfinite(raw)):
            raise ValueError("Nonfinite trace samples")
        raise ValueError("Samples overflow Sauce float32 trace storage")
    return samples


class _FrequencyTransform:
    """Reuse a bounded transform plan across trace batches."""

    def __init__(
        self, nt: int, frequencies: np.ndarray, dt: float, t0: float, budget: int
    ) -> None:
        self.frequencies = frequencies
        self.nt, self.dt, self.t0 = nt, dt, t0
        self.bins = np.rint(frequencies * nt * dt).astype(np.int64)
        bin_frequency = self.bins / (nt * dt)
        tolerance = np.maximum(1e-8, 1e-8 * np.maximum(1, np.abs(frequencies)))
        self.fft = len(frequencies) >= 4 and np.all(
            np.abs(bin_frequency - frequencies) <= tolerance
        )
        self.width = max(1, min(256, budget // max(1, nt * 32)))
        self.kernels = None
        if not self.fft and len(frequencies) <= self.width:
            self.kernels = self._kernels(frequencies)

    def _kernels(self, frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = self.t0 + np.arange(self.nt, dtype=np.float64) * self.dt
        phase = (-2 * np.pi * t[:, None]) * frequencies[None, :]
        return np.asfortranarray(np.cos(phase)), np.asfortranarray(np.sin(phase))

    def blocks(self, samples: np.ndarray) -> Iterator[tuple[int, int, np.ndarray]]:
        if self.fft:
            from scipy.fft import rfft

            spectrum = rfft(samples, axis=1)
        else:
            from scipy.linalg.blas import dgemm

            work = np.asfortranarray(samples, dtype=np.float64)
        for start in range(0, len(self.frequencies), self.width):
            stop = min(start + self.width, len(self.frequencies))
            frequencies = self.frequencies[start:stop]
            if self.fft:
                values = spectrum[:, self.bins[start:stop]] * np.exp(
                    -2j * np.pi * frequencies * self.t0
                )
            else:
                cosine, sine = (
                    self.kernels
                    if self.kernels is not None
                    else self._kernels(frequencies)
                )
                real = dgemm(1.0, work, cosine)
                imag = dgemm(1.0, work, sine)
                values = real + 1j * imag
            yield start, stop, _checked_samples(values, "frequency")


def _write_store(
    reader: _ArrayTraces | _SegyTraces,
    path: Path,
    *,
    batch_bytes: int,
    component_map: Mapping[str, int] | None,
    frequencies: Sequence[float] | np.ndarray | None,
) -> None:
    ntrace, nsample = reader.ntrace, reader.nsample
    # Account for source samples, float64/complex128 conversion, and transform work.
    batch = min(65536, max(1, batch_bytes // max(1, nsample * 32)))
    matrix = reader.matrix if isinstance(reader, _ArrayTraces) else None
    sample_major = isinstance(reader, _ArrayTraces) and reader.sample_major
    if sample_major:
        assert matrix is not None
    sample_tile = (
        max(1, batch_bytes // max(1, ntrace * (matrix.dtype.itemsize + 8)))
        if sample_major and matrix is not None
        else nsample
    )
    strings = h5py.string_dtype("utf-8")
    mapping = (
        dict(component_map)
        if component_map is not None
        else {name: i + 1 for i, name in enumerate(reader.component_names())}
    )
    if mapping:
        _ids(list(mapping.values()), "component_map")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("component_map values must be unique")
    with h5py.File(path, "w") as h5:
        h5["survey/schema_version"] = np.asarray(
            ["fs_seismic_trace_store_v1"], dtype=strings
        )
        h5["survey/layout_kind"] = np.asarray(["sparse_trace_v1"], dtype=strings)
        h5.attrs["conversion"] = json.dumps(reader.provenance, sort_keys=True)
        table = h5.require_group("survey/traces")
        columns = {
            name: table.create_dataset(name, shape=(ntrace,), dtype="i4")
            for name in (
                "trace_id",
                "source_id",
                "receiver_id",
                "component_id",
                "component",
                "active",
            )
        }
        chunk_samples = min(nsample, sample_tile) if sample_major else nsample
        chunk_rows = min(ntrace, batch, max(1, 1024**2 // (chunk_samples * 8)))
        if reader.domain == "time":
            values = h5.create_dataset(
                "data/time/value",
                shape=(ntrace, nsample),
                dtype="f4",
                chunks=(chunk_rows, min(nsample, sample_tile, 262144)),
            )
            values.attrs.update(dt=reader.dt, t0=reader.t0)
            values.attrs["dims"] = np.asarray(["trace", "time"], dtype=strings)
            h5["data/time/dt"], h5["data/time/t0"] = reader.dt, reader.t0
        else:
            values = h5.create_dataset(
                "data/frequency/value",
                shape=(ntrace, nsample, 2),
                dtype="f4",
                chunks=(chunk_rows, min(nsample, 131072), 2),
            )
            h5["data/frequency/frequency"] = reader.frequency
            h5["data/frequency/laplace"] = reader.laplace
            h5["data/frequency/time_power"] = np.asarray(
                reader.time_power, dtype=np.int32
            )
        cached = None
        if frequencies is not None:
            if reader.domain != "time":
                raise ValueError(
                    "frequencies cache preparation only applies to time inputs"
                )
            assert reader.dt is not None and reader.t0 is not None
            frequencies = _axis(frequencies, "frequencies")
            if (
                np.any(frequencies < 0)
                or np.any(frequencies > 0.5 / reader.dt)
                or len(np.unique(frequencies)) != len(frequencies)
            ):
                raise ValueError(
                    "Cache frequencies must be unique and within [0, Nyquist]"
                )
            cache_rows = min(ntrace, max(1024, min(batch, 8192)))
            cache_bytes = min(
                batch_bytes // 4,
                max(1024**2, cache_rows * min(len(frequencies), 256) * 16),
            )
            cached = h5.create_dataset(
                "data/frequency/value",
                shape=(ntrace, len(frequencies), 2),
                dtype="f4",
                chunks=(cache_rows, 1, 2),
                rdcc_nbytes=cache_bytes,
                rdcc_nslots=10007,
            )
            h5["data/frequency/frequency"] = frequencies
            h5["data/frequency/laplace"] = np.zeros(len(frequencies))
            h5["data/frequency/time_power"] = np.zeros(len(frequencies), dtype=np.int32)
        transform = None
        if cached is not None:
            assert (
                frequencies is not None
                and reader.dt is not None
                and reader.t0 is not None
            )
            transform = _FrequencyTransform(
                nsample, np.asarray(frequencies), reader.dt, reader.t0, batch_bytes // 2
            )
        if sample_major:
            assert isinstance(reader, _ArrayTraces)
            for first in range(0, nsample, sample_tile):
                last = min(first + sample_tile, nsample)
                tile = _checked_samples(reader.read_sample_tile(first, last), "time")
                values[:, first:last] = tile
                del tile
        for start in range(0, ntrace, batch):
            stop = min(ntrace, start + batch)
            metadata = reader.metadata(start, stop)
            names = np.asarray(metadata.pop("component_name")).astype(str)
            unique_names, inverse = np.unique(names, return_inverse=True)
            for name in unique_names:
                if name not in mapping:
                    raise ValueError(f"Missing component_map entry for {name!r}")
            components = np.asarray(
                [mapping[name] for name in unique_names], dtype=np.int32
            )[inverse]
            metadata.update(
                trace_id=np.arange(start + 1, stop + 1),
                component_id=components,
                component=components,
                active=np.ones(stop - start, dtype=np.int32),
            )
            for name, column in columns.items():
                column[start:stop] = metadata[name]
            if sample_major:
                if cached is None:
                    continue
                samples = values[start:stop]
            else:
                samples = _checked_samples(reader.read(start, stop), reader.domain)
                if reader.domain == "time":
                    values[start:stop] = samples
                else:
                    packed = (
                        np.ascontiguousarray(samples)
                        .view(np.float32)
                        .reshape(stop - start, nsample, 2)
                    )
                    values[start:stop] = packed
            if transform is not None:
                assert cached is not None
                for first, last, transformed in transform.blocks(samples):
                    packed = (
                        np.ascontiguousarray(transformed)
                        .view(np.float32)
                        .reshape(stop - start, last - first, 2)
                    )
                    cached[start:stop, first:last] = packed
        h5["survey/components/component_id"] = np.asarray(
            list(mapping.values()), dtype=np.int32
        )
        h5["survey/components/component"] = np.asarray(
            list(mapping.values()), dtype=np.int32
        )
        h5["survey/components/component_name"] = np.asarray(
            list(mapping), dtype=strings
        )


def convert_traces(
    data: Any,
    output: str | Path,
    *,
    dims: Sequence[str] | None = None,
    variable: str | None = None,
    dt: float | None = None,
    t0: float | None = None,
    source_ids: Any = None,
    receiver_ids: Any = None,
    component: str | None = None,
    component_map: Mapping[str, int] | None = None,
    frequencies: Sequence[float] | None = None,
    batch_bytes: int = 64 * 1024**2,
    segy_headers: Mapping[str, int] | None = None,
    endian: str = "big",
    overwrite: bool = False,
) -> Path:
    """Write a Sauce observed store from SEG-Y, NPY, NumPy, or xarray.

    Arrays default to (trace, time); specify ``dims`` for dense NumPy cubes.
    Xarray dimensions are source/receiver/component or trace plus time/frequency.
    IDs come from coordinates, then attributes (source_id), then one-based indices.
    Explicit ID arrays follow flattened trace order. Supply a component name when
    absent from xarray metadata, and component_map to match solver output numbers.

    Time and frequency coordinates use seconds and Hz. NPY payloads stream in
    bounded blocks; uncommon strided layouts use temporary memory maps.
    ``frequencies`` optionally prepares an unnormalized Fourier cache from time
    samples; the original time data remains available for other coordinates.
    batch_bytes sets the working-buffer target, with a minimum of one trace
    (one time slice for sample-major input). Library workspaces are additional.
    Fourier-bin caches use FFTs; off-grid frequencies use blocked matrix products.

    SEG-Y assumes fixed-length traces and uniform sampling. Header byte positions
    default to FieldRecord, TraceNumber, sample interval, and recording delay.
    Override with segy_headers when IDs occupy other fields. Geometry and physical
    units must be configured separately in the Sauce acquisition; this converter
    preserves identifiers and sample amplitudes without geometry inference.

    Output is published atomically only after successful conversion; an existing
    file is preserved unless overwrite=True. Zarr input is not supported here.
    """
    if not isinstance(batch_bytes, int) or batch_bytes <= 0:
        raise ValueError("batch_bytes must be a positive integer")
    output = Path(output).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    reader: _ArrayTraces | _SegyTraces | None = None
    temp = None
    source = None
    try:
        if isinstance(data, (str, Path)):
            source = Path(data).expanduser().resolve()
            if source == output:
                raise ValueError("Input and output must be different files")
            if source.suffix.lower() in {".sgy", ".segy"}:
                if any(value is not None for value in (dims, variable, dt, t0)):
                    raise ValueError("Use segy_headers for SEG-Y sampling")
                reader = _SegyTraces(
                    source,
                    component=component,
                    headers=segy_headers,
                    endian=endian,
                    source_ids=source_ids,
                    receiver_ids=receiver_ids,
                )
            elif source.suffix.lower() == ".npy":
                data = np.load(source, mmap_mode="r", allow_pickle=False)
            else:
                raise ValueError("Input files must be .sgy, .segy, or .npy")
        if reader is None:
            if segy_headers is not None:
                raise ValueError("segy_headers only applies to SEG-Y inputs")
            reader = _ArrayTraces(
                data,
                dims=dims,
                variable=variable,
                dt=dt,
                t0=t0,
                source_ids=source_ids,
                receiver_ids=receiver_ids,
                component=component,
            )
        if source is not None and source.suffix.lower() == ".npy":
            assert isinstance(reader, _ArrayTraces)
            reader.provenance.update(format="npy", path=str(source))
            reader.npy_file = source
            reader.npy_offset = data.offset
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".partial", dir=output.parent
        )
        os.close(descriptor)
        temp = Path(filename)
        _write_store(
            reader,
            temp,
            batch_bytes=batch_bytes,
            component_map=component_map,
            frequencies=frequencies,
        )
        if overwrite:
            os.replace(temp, output)
        else:
            os.link(temp, output)
        return output
    finally:
        if isinstance(reader, _SegyTraces):
            reader.file.close()
        if temp is not None:
            temp.unlink(missing_ok=True)
