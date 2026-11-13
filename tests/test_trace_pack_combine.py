"""Regression coverage for combining multi-frequency native trace segments."""

import h5py
import numpy as np
import pytest

from frequensolve.seismic.trace_pack import (
    TracePackEntry,
    TracePackManifest,
    TracePackSegment,
)
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.jobs.artifacts import TraceManifest


def segment(tmp_path, name, frequencies, offset=0):
    path = tmp_path / f"{name}.h5"
    strings = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h:
        h["survey/receivers/coordinates"] = [[0.0, 1.0], [0.0, 0.0]]
        for key, value in {
            "layout_kind": "indexed_frequency_trace_v1",
            "data_root": "/trace_data",
            "schema_version": "fs-trace-index-1",
            "dataset_number_format": "%06d",
        }.items():
            h.create_dataset(f"trace_index/{key}", data=[value], dtype=strings)
        for key, values in {
            "frequency": [f.real for f in frequencies],
            "laplace": [f.imag for f in frequencies],
            "dataset_number": np.arange(1, len(frequencies) + 1),
        }.items():
            h[f"trace_index/{key}"] = values
        numbers, paths, originals = [], [], []
        for n, f in enumerate(frequencies, 1):
            for group in ["seabed", "surface"]:
                p = f"/trace_data/{group}/{n:06d}"
                h[p] = np.full((2, 1, 3, 2), offset + n, dtype="f4")
                h[p + "_rhs_amplitude_scale"] = [offset + n, offset + n + 1]
                h[p].attrs.create(
                    "value_convention_rhs_amplitude_scale_dataset",
                    [p + "_rhs_amplitude_scale"],
                    dtype=strings,
                )
                numbers.append(n)
                paths.append(p)
                originals.append("/" + group)
        h["trace_index/datasets/dataset_number"] = numbers
        h.create_dataset("trace_index/datasets/packed_path", data=paths, dtype=strings)
        h.create_dataset(
            "trace_index/datasets/source_path", data=originals, dtype=strings
        )
    pack = TracePackManifest(
        None,
        name,
        "traces",
        "traces",
        "simulated_traces",
        "/trace_data",
        "/trace_index",
        (
            TracePackSegment(
                "segment",
                path.name,
                path,
                "fs-traces-packed-1",
                name,
                path.stat().st_size,
                None,
            ),
        ),
        tuple(
            TracePackEntry(n, f, "segment", n, path.name, name)
            for n, f in enumerate(frequencies, 1)
        ),
    )
    return TraceManifest(
        [path],
        {n: f.real for n, f in enumerate(frequencies, 1)},
        ["seabed", "surface"],
        tmp_path / "sim.json",
        tmp_path,
        tmp_path,
        laplace={n: f.imag for n, f in enumerate(frequencies, 1)},
        pack=pack,
    )


def test_merge_preserves_damping_and_all_packed_datasets(tmp_path):
    a = segment(tmp_path, "a", [1 - 0.025j, 2 - 0.025j])
    b = segment(tmp_path, "b", [1 - 0.005j, 2 - 0.005j], 10)
    merged = TraceDataset(TraceManifest.combine([a, b], duplicate="error"))
    assert len(merged.manifest.frequencies) == 4
    out = merged.export_packed(tmp_path / "combined.h5")
    with h5py.File(out) as h:
        assert set(zip(h["frequency"][()], h["laplace"][()])) == {
            (1, -0.025),
            (1, -0.005),
            (2, -0.025),
            (2, -0.005),
        }
        for n, e in enumerate(merged.metadata["packed_entries"], 1):
            with h5py.File(e["path"]) as original:
                for group in ["seabed", "surface"]:
                    trace_path = f"/trace_data/{group}/{n:06d}"
                    companion = h[trace_path].attrs[
                        "value_convention_rhs_amplitude_scale_dataset"
                    ][0]
                    assert companion == trace_path + "_rhs_amplitude_scale"
                    assert np.array_equal(
                        h[companion][()],
                        original[
                            f"trace_data/{group}/{e['dataset_number']:06d}_rhs_amplitude_scale"
                        ][()],
                    )
                    for suffix in ["", "_rhs_amplitude_scale"]:
                        assert np.array_equal(
                            h[f"trace_data/{group}/{n:06d}{suffix}"][()],
                            original[
                                f"trace_data/{group}/{e['dataset_number']:06d}{suffix}"
                            ][()],
                        )
        assert len(h["trace_index/datasets/packed_path"]) == 8
    with pytest.raises(FileExistsError):
        merged.export_packed(out)


def test_packed_duplicates_and_incompatible_survey_are_rejected(tmp_path):
    a = segment(tmp_path, "a", [1 - 0.025j, 2 - 0.025j])
    b = segment(tmp_path, "b", [2 - 0.025j, 3 - 0.025j], 10)
    with pytest.raises(ValueError, match="Duplicate"):
        TraceManifest.combine([a, b], duplicate="error")
    for policy, expected in [("first", "a.h5"), ("last", "b.h5")]:
        merged = TraceDataset(TraceManifest.combine([a, b], duplicate=policy))
        assert len(merged.manifest.frequencies) == 3
        assert merged.metadata["packed_entries"][1]["path"].endswith(expected)
    with h5py.File(b.files[0], "r+") as h:
        h["survey/receivers/coordinates"][0, 0] = 9
    with pytest.raises(ValueError, match="surveys differ"):
        merged.export_packed(tmp_path / "bad.h5")
    assert not (tmp_path / "bad.h5").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_selected_group_export_and_standalone_reexport(tmp_path):
    from dataclasses import replace

    manifest = segment(tmp_path, "source", [1 - 0.02j, 2 - 0.02j])
    strings = h5py.string_dtype("utf-8")
    with h5py.File(manifest.files[0], "r+") as h:
        for name in ["seabed", "surface"]:
            h[f"survey/receiver_groups/{name}/receivers/receiver_id"] = [1, 2, 3]
        h.create_dataset(
            "survey/receiver_groups/_catalog/group_name",
            data=["seabed", "surface"],
            dtype=strings,
        )
        h.create_dataset(
            "survey/receiver_groups/_catalog/dataset_path",
            data=["/seabed", "/surface"],
            dtype=strings,
        )
    selected = TraceDataset(replace(manifest, groups=["surface"]))
    first = selected.export_packed(tmp_path / "selected.h5")
    reopened = TraceDataset.open(first)
    assert reopened.manifest.groups == ["surface"]
    second = reopened.export_packed(tmp_path / "again.h5")
    with h5py.File(second) as h:
        assert list(h["trace_data"]) == ["surface"]
        assert list(h["survey/receiver_groups"]) == ["_catalog", "surface"]
        assert h["survey/receiver_groups/_catalog/group_name"].asstr()[()].tolist() == [
            "surface"
        ]
        assert h["trace_index/datasets/source_path"].asstr()[()].tolist() == [
            "/surface",
            "/surface",
        ]
        np.testing.assert_array_equal(h["laplace"][()], [-0.02, -0.02])
        np.testing.assert_array_equal(h["trace_data/surface/000002"][()], 2)
    combined = TraceDataset.combine(
        [reopened, TraceDataset.open(second)], duplicate="first"
    )
    assert combined.manifest.pack is not None
