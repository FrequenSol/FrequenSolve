"""Measure trace conversion throughput and peak RSS in a fresh process.

Prepare a fixture, then run each measurement as a separate invocation::

    python -m benchmarks.trace_conversion --directory /tmp/trace-bench --prepare
    python -m benchmarks.trace_conversion --directory /tmp/trace-bench
    python -m benchmarks.trace_conversion --directory /tmp/trace-bench --cache 32

Use --kind segy or time-first to test other layouts. --implementation can point
at a saved trace_conversion.py for before/after comparisons. Measurements include
conversion and fsync, exclude fixture generation/imports/validation, and report
input payload MiB/s. OS/device caches affect results; these are local throughput
measurements, not cold-storage or network performance guarantees.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np


def prepare(path, kind, rows, samples):
    rng = np.random.default_rng(42)
    if kind == "segy":
        import segyio

        spec = segyio.spec()
        spec.format, spec.sorting, spec.tracecount = 5, 2, rows
        spec.samples = np.arange(samples)
        with segyio.create(str(path), spec) as file:
            file.bin[segyio.BinField.Interval] = 1000
            for start in range(0, rows, 256):
                stop = min(start + 256, rows)
                file.trace.raw[start:stop] = rng.normal(
                    size=(stop - start, samples)
                ).astype("f4")
                for row in range(start, stop):
                    file.header[row] = {
                        segyio.TraceField.FieldRecord: 1,
                        segyio.TraceField.TraceNumber: row + 1,
                        segyio.TraceField.TRACE_SAMPLE_INTERVAL: 1000,
                    }
    else:
        shape = (samples, rows) if kind == "time-first" else (rows, samples)
        data = np.lib.format.open_memmap(path, mode="w+", dtype="f4", shape=shape)
        block_rows = max(1, 8 * 1024**2 // (shape[1] * 8))
        for start in range(0, shape[0], block_rows):
            stop = min(start + block_rows, shape[0])
            data[start:stop] = rng.normal(size=(stop - start, shape[1])).astype("f4")
        data.flush()


def validate(source, output, kind, rows, samples, frequencies):
    selected = [0, rows // 2, rows - 1]
    if kind == "segy":
        import segyio

        with segyio.open(str(source), ignore_geometry=True) as file:
            expected = np.stack([file.trace[row].copy() for row in selected])
    else:
        data = np.load(source, mmap_mode="r")
        expected = data[:, selected].T if kind == "time-first" else data[selected]
    with h5py.File(output) as file:
        assert file["data/time/value"].shape == (rows, samples)
        np.testing.assert_array_equal(file["data/time/value"][selected], expected)
        if frequencies is not None:
            time_axis = np.arange(samples) * 0.001
            count = min(4, len(frequencies))
            reference = np.einsum(
                "rt,tf->rf",
                expected.astype(np.float64),
                np.exp(-2j * np.pi * time_axis[:, None] * frequencies[:count]),
                optimize=False,
            )
            actual = file["data/frequency/value"][selected, :count]
            np.testing.assert_allclose(
                actual[..., 0] + 1j * actual[..., 1], reference, rtol=1e-5, atol=1e-4
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--kind", choices=("npy", "segy", "time-first"), default="npy")
    parser.add_argument("--mib", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument(
        "--cache", type=int, default=0, help="Number of prepared frequencies"
    )
    parser.add_argument(
        "--off-grid",
        action="store_true",
        help="Use arbitrary frequencies instead of Fourier bins",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Create/replace the benchmark input, without timing",
    )
    parser.add_argument(
        "--implementation", type=Path, help="Saved converter module to compare"
    )
    args = parser.parse_args()
    if args.mib < 1 or args.samples < 2 or not 0 <= args.cache < args.samples // 2:
        parser.error("Use positive MiB, samples >= 2, and cache < samples/2")
    rows = args.mib * 1024**2 // (args.samples * 4)
    if rows < 3:
        parser.error("Fixture must have at least three traces")
    args.directory.mkdir(parents=True, exist_ok=True)
    suffix = ".sgy" if args.kind == "segy" else ".npy"
    source = args.directory / f"{args.kind}-{args.mib}-{args.samples}{suffix}"
    if args.prepare:
        prepare(source, args.kind, rows, args.samples)
        print(source)
        return
    if not source.is_file():
        parser.error("Run with --prepare first")
    if args.implementation:
        spec = importlib.util.spec_from_file_location(
            "benchmark_converter", args.implementation
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        convert = module.convert_traces
    else:
        from frequensolve.seismic import convert_traces as convert
    options = {"component": "p", "overwrite": True}
    if args.kind != "segy":
        options["dt"] = 0.001
    if args.kind == "time-first":
        options["dims"] = ("time", "receiver")
    frequencies = None
    if args.cache:
        frequencies = (
            np.arange(1, args.cache + 1) + (0.123 if args.off_grid else 0)
        ) / (args.samples * 0.001)
        options["frequencies"] = frequencies
    descriptor, name = tempfile.mkstemp(suffix=".h5", dir=args.directory)
    os.close(descriptor)
    output = Path(name)
    try:
        start = time.perf_counter()
        convert(source, output, **options)
        with output.open("rb") as file:
            os.fsync(file.fileno())
        elapsed = time.perf_counter() - start
        peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2 if sys.platform == "darwin" else 1024
        )
        validate(source, output, args.kind, rows, args.samples, frequencies)
        print(
            json.dumps(
                {
                    "kind": args.kind,
                    "input_mib": rows * args.samples * 4 / 1024**2,
                    "cache_frequencies": args.cache,
                    "off_grid": args.off_grid,
                    "seconds": elapsed,
                    "mib_per_second": rows * args.samples * 4 / 1024**2 / elapsed,
                    "peak_rss_mib": peak_mib,
                    "validation": "passed",
                }
            )
        )
    finally:
        output.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
