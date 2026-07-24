import json
import warnings
from pathlib import Path

import h5py
import numpy as np

from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.jobs.artifacts import TraceManifest


def _write_accumulated_packed_product(path: Path) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    frequencies = np.array([0.2, 0.4, 0.1, 0.2, 0.3, 0.4])
    laplace = np.array([-0.2, -0.2, -0.1, -0.1, -0.1, -0.1])
    task_ids = np.array([1, 2, 1, 2, 3, 4], dtype=np.int32)
    dataset_numbers = np.arange(1, 7, dtype=np.int32)

    with h5py.File(path, "w") as h5:
        h5.create_dataset("frequency", data=frequencies)
        h5.create_dataset("laplace", data=laplace)
        h5.create_dataset("task_id", data=task_ids)
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["indexed_frequency_trace_v1"], dtype=string_dtype),
        )

        catalog = h5.require_group("survey/receiver_groups/_catalog")
        catalog.create_dataset(
            "group_name",
            data=np.array(["surface"], dtype=string_dtype),
        )
        catalog.create_dataset(
            "dataset_path",
            data=np.array(["/surface"], dtype=string_dtype),
        )
        catalog.create_dataset(
            "layout_kind",
            data=np.array(["dense_trace_v1"], dtype=string_dtype),
        )
        trace_group = h5.require_group("survey/receiver_groups/surface/traces")
        trace_group.create_dataset("receiver_id", data=np.array([101]))
        trace_group.create_dataset("source_id", data=np.array([7]))
        trace_group.create_dataset(
            "component_name",
            data=np.array(["p"], dtype=string_dtype),
        )

        h5.require_group("trace_index/datasets")
        h5.create_dataset(
            "trace_index/layout_kind",
            data=np.array(["indexed_frequency_trace_v1"], dtype=string_dtype),
        )
        h5.create_dataset("trace_index/dataset_number", data=dataset_numbers)
        h5.create_dataset("trace_index/frequency", data=frequencies)
        h5.create_dataset("trace_index/laplace", data=laplace)
        h5.create_dataset("trace_index/task_id", data=task_ids)
        h5.create_dataset(
            "trace_index/datasets/dataset_number",
            data=dataset_numbers,
        )
        h5.create_dataset(
            "trace_index/datasets/source_path",
            data=np.array(["/surface"] * 6, dtype=string_dtype),
        )
        h5.create_dataset(
            "trace_index/datasets/packed_path",
            data=np.array(
                [f"/trace_data/surface/{number:06d}" for number in dataset_numbers],
                dtype=string_dtype,
            ),
        )

        for number, value in zip(dataset_numbers, [20.0, 40.0, 1.0, 2.0, 3.0, 4.0]):
            dataset = h5.create_dataset(
                f"trace_data/surface/{number:06d}",
                data=np.array([[[[value, 0.0]]]], dtype=np.float32),
            )
            dataset.attrs["dims"] = ["receiver", "component", "shot"]
            dataset.attrs["receiver"] = np.array([101])
            dataset.attrs["component"] = np.array(["p"], dtype=string_dtype)
            dataset.attrs["shot"] = np.array([7])


def _trace_manifest(tmp_path: Path) -> TraceManifest:
    trace_dir = tmp_path / "results" / "traces"
    return TraceManifest(
        files=[trace_dir / f"traces_{task}.h5" for task in range(1, 5)],
        frequencies={1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
        laplace={1: -0.1, 2: -0.1, 3: -0.1, 4: -0.1},
        groups=["surface"],
        simulation=tmp_path / "simulation.json",
        result_path=tmp_path / "results",
        output_path=trace_dir,
        project_path=tmp_path,
    )


def test_accumulated_packed_runs_preserve_current_frequency_laplace_rows(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    packed = trace_dir / "traces.h5"
    _write_accumulated_packed_product(packed)

    current_rows = [
        {
            "dataset_number": task + 2,
            "task_id": task,
            "frequency": task / 10,
            "laplace": -0.1,
            "status": "packed",
        }
        for task in range(1, 5)
    ]
    stale_rows = [
        {
            "dataset_number": task,
            "task_id": task,
            "frequency": task / 5,
            "laplace": -0.2,
            "status": "packed",
        }
        for task in range(1, 3)
    ]
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "traces/traces.h5",
                },
                "frequencies": [*current_rows, *stale_rows],
            }
        )
    )
    manifest = _trace_manifest(tmp_path)

    assert manifest.missing_packed_frequencies == {}
    assert manifest.packed_complete
    assert len(manifest.packed_frequencies) == 6

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        traces = TraceDataset.from_manifest(manifest)
        frequency_data = traces.fd("surface", "p", source=7)

    assert not caught
    assert traces.manifest.frequencies == manifest.frequencies
    assert frequency_data.frequency.values.tolist() == [0.1, 0.2, 0.3, 0.4]
    assert frequency_data.values.real[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_stale_laplace_row_does_not_cover_current_frequency(tmp_path):
    trace_dir = tmp_path / "results" / "traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "traces.h5").touch()
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "packed": {"relative_path": "traces/traces.h5"},
                "frequencies": [
                    {
                        "task_id": 1,
                        "frequency": 0.1,
                        "laplace": -0.2,
                    }
                ],
            }
        )
    )
    manifest = _trace_manifest(tmp_path)

    assert manifest.missing_packed_frequencies == {
        1: 0.1,
        2: 0.2,
        3: 0.3,
        4: 0.4,
    }
    assert not manifest.packed_complete
