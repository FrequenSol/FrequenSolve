import json

import h5py
import numpy as np
import pytest

from frequensolve.seismic.trace_pack import TracePackManifest
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.artifact_catalog import load_artifact_catalog
from frequensolve.simulation.artifact_contract import (
    ArtifactContractError,
    ArtifactRequest,
)
from frequensolve.simulation.jobs.artifacts import RunMetadata, TraceManifest


def _write_segment(path, *, value, frequency, dataset_number):
    path.parent.mkdir(parents=True, exist_ok=True)
    string = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("frequency", data=np.asarray([frequency]))
        h5.create_dataset("laplace", data=np.asarray([0.0]))
        index = h5.require_group("trace_index")
        index.create_dataset(
            "layout_kind", data=np.asarray(["indexed_frequency_trace_v1"], dtype=string)
        )
        index.create_dataset(
            "dataset_number", data=np.asarray([dataset_number], dtype=np.int64)
        )
        index.create_dataset("frequency", data=np.asarray([frequency]))
        index.create_dataset("laplace", data=np.asarray([0.0]))
        datasets = index.require_group("datasets")
        datasets.create_dataset(
            "dataset_number", data=np.asarray([dataset_number], dtype=np.int64)
        )
        datasets.create_dataset(
            "source_path", data=np.asarray(["/surface"], dtype=string)
        )
        datasets.create_dataset(
            "packed_path",
            data=np.asarray(
                [f"/trace_data/surface/d{dataset_number:08d}"], dtype=string
            ),
        )
        survey = h5.require_group("survey")
        sources = survey.require_group("sources")
        sources.create_dataset("source_id", data=np.asarray([1], dtype=np.int64))
        receivers = survey.require_group("receiver_groups/surface/receivers")
        receivers.create_dataset("receiver_id", data=np.asarray([10, 11]))
        components = survey.require_group("receiver_groups/surface/components")
        components.create_dataset(
            "component_name", data=np.asarray(["p"], dtype=string)
        )
        catalog = survey.require_group("receiver_groups/_catalog")
        catalog.create_dataset("group_name", data=np.asarray(["surface"], dtype=string))
        catalog.create_dataset(
            "dataset_path", data=np.asarray(["/surface"], dtype=string)
        )
        data = np.zeros((2, 1, 1, 2), dtype=np.float32)
        data[:, 0, 0, 0] = value
        dset = h5.create_dataset(f"trace_data/surface/d{dataset_number:08d}", data=data)
        dset.attrs["dims"] = np.asarray(
            ["receiver", "component", "shot", "complex"], dtype=string
        )


def _pack_catalog(tmp_path, *, dataset_numbers=(1, 2)):
    result = tmp_path / "results"
    segment_rows = []
    segment_artifacts = []
    entries = []
    for task, (frequency, dataset_number) in enumerate(
        zip((1.0, 2.0), dataset_numbers), start=1
    ):
        relative = f"traces/segments/segment-{task}.h5"
        path = result / relative
        _write_segment(
            path,
            value=float(task),
            frequency=frequency,
            dataset_number=dataset_number,
        )
        segment_id = f"traces:segment:g{task}"
        segment_rows.append(
            {
                "id": segment_id,
                "path": relative,
                "schema": "fs-traces-packed-1",
                "generation": f"g{task}",
                "bytes": path.stat().st_size,
            }
        )
        segment_artifacts.append(
            {
                "id": segment_id,
                "role": "trace_data",
                "representation": "packed_hdf5",
                "schema": "fs-traces-packed-1",
                "path": relative,
                "retention": "durable",
                "generation": f"g{task}",
                "bytes": path.stat().st_size,
            }
        )
        entries.append(
            {
                "task": task,
                "frequency": {"real": frequency, "imag": 0.0},
                "segment_id": segment_id,
                "dataset_number": dataset_number,
                "source_path": f"traces/shards/task-{task}.h5",
                "source_generation": f"source-{task}",
            }
        )
    manifest_relative = "traces/manifest.json"
    manifest_path = result / manifest_relative
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-2",
                "generation": "pack-1",
                "family": {
                    "id": "traces",
                    "kind": "traces",
                    "role": "simulated_traces",
                },
                "trace_data_root": "/trace_data",
                "index_path": "/trace_index",
                "segments": segment_rows,
                "entries": entries,
            }
        )
    )
    manifest_artifact = {
        "id": "traces",
        "role": "simulated_traces",
        "representation": "packed_manifest",
        "schema": "fs-trace-manifest-2",
        "path": manifest_relative,
        "retention": "durable",
        "generation": "pack-1",
        "bytes": manifest_path.stat().st_size,
        "dependencies": [row["id"] for row in segment_rows],
    }
    operation = result / "_fs_run/operations/pack/result.json"
    operation.parent.mkdir(parents=True, exist_ok=True)
    operation.write_text(
        json.dumps(
            {
                "schema": "fs-operation-result-1",
                "operation": {"name": "pack", "generation": "pack-1"},
                "fingerprints": {
                    "job": "1" * 64,
                    "simulation": "2" * 64,
                    "outputs": "3" * 64,
                },
                "status": {"state": "success", "code": 0},
                "artifacts": [*segment_artifacts, manifest_artifact],
            }
        )
    )
    catalog = load_artifact_catalog(result, tasks=(), operations=("pack",))
    artifact = catalog.require_one(
        ArtifactRequest(
            id="traces",
            role="simulated_traces",
            representations=("packed_manifest",),
        ),
        operation="pack",
    )
    return result, catalog, artifact


def test_trace_pack_manifest_validates_without_opening_segment_payloads(
    tmp_path, monkeypatch
):
    _result, catalog, artifact = _pack_catalog(tmp_path)
    original = h5py.File

    def reject_hdf(*args, **kwargs):
        raise AssertionError("selection must not open segment HDF5")

    monkeypatch.setattr(h5py, "File", reject_hdf)
    pack = TracePackManifest.read(artifact, catalog=catalog)
    monkeypatch.setattr(h5py, "File", original)

    assert [entry.task for entry in pack.entries_for((1.0, 2.0))] == [1, 2]
    assert [segment.id for segment in pack.selected_segments(pack.entries)] == [
        "traces:segment:g1",
        "traces:segment:g2",
    ]


def test_trace_pack_manifest_rejects_zero_dataset_numbers(tmp_path):
    _result, catalog, artifact = _pack_catalog(tmp_path, dataset_numbers=(0, 0))
    with pytest.raises(ArtifactContractError, match="dataset_number"):
        TracePackManifest.read(artifact, catalog=catalog)


def test_trace_pack_manifest_rejects_operation_dependency_mismatch(tmp_path):
    _result, catalog, artifact = _pack_catalog(tmp_path)
    object.__setattr__(artifact, "dependencies", ("traces:segment:g1",))

    with pytest.raises(ArtifactContractError, match="dependencies"):
        TracePackManifest.read(artifact, catalog=catalog)


def test_trace_dataset_reads_only_explicit_multi_segment_entries(tmp_path):
    result, catalog, artifact = _pack_catalog(tmp_path)
    pack = TracePackManifest.read(artifact, catalog=catalog)
    manifest = TraceManifest(
        files=[segment.path for segment in pack.segments],
        frequencies={1: 1.0, 2: 2.0},
        groups=["surface"],
        simulation=tmp_path / "simulation.json",
        result_path=result,
        output_path=result / "traces",
        project_path=tmp_path,
        laplace={1: 0.0, 2: 0.0},
        artifacts=[artifact, *(segment.artifact for segment in pack.segments)],
        run=RunMetadata(result_path=result),
        pack=pack,
    )

    traces = TraceDataset.from_manifest(manifest)
    gather = traces.fd("surface", "p", source=1).compute()

    assert gather.coords["frequency"].values.tolist() == [1.0, 2.0]
    assert gather.real.values.tolist() == [[1.0, 1.0], [2.0, 2.0]]


def test_trace_manifest_resolves_pack_only_through_fixed_operation_result(tmp_path):
    result, _catalog, _artifact = _pack_catalog(tmp_path)

    pack = TraceManifest._receiver_pack(result, (1.0, 2.0))

    assert pack is not None
    assert [segment.relative_path for segment in pack.segments] == [
        "traces/segments/segment-1.h5",
        "traces/segments/segment-2.h5",
    ]

    assert (
        TraceManifest._receiver_pack(
            result,
            (1.0, 2.0),
            fingerprints={"job": "9" * 64},
        )
        is None
    )


def test_sampled_wavefield_families_read_explicit_packed_task_datasets(tmp_path):
    result = tmp_path / "results"
    for task, frequency in enumerate((1.0, 2.0), 1):
        artifacts = []
        for offset, group in enumerate(("surface", "other")):
            path = result / f"opaque/{group}-{task}.h5"
            _write_segment(
                path,
                value=task + 10 * offset,
                frequency=frequency,
                dataset_number=7 + task,
            )
            if group != "surface":
                string = h5py.string_dtype("utf-8")
                with h5py.File(path, "a") as h5:
                    h5.move("trace_data/surface", "trace_data/other")
                    h5.move(
                        "survey/receiver_groups/surface", "survey/receiver_groups/other"
                    )
                    for name, value in {
                        "trace_index/datasets/source_path": "/other",
                        "trace_index/datasets/packed_path": f"/trace_data/other/d{7 + task:08d}",
                        "survey/receiver_groups/_catalog/group_name": "other",
                        "survey/receiver_groups/_catalog/dataset_path": "/other",
                    }.items():
                        del h5[name]
                        h5.create_dataset(name, data=np.asarray([value], dtype=string))
            artifacts.append(
                {
                    "id": f"wavefield:fields/{group}",
                    "role": "sampled_wavefield",
                    "representation": "packed_trace",
                    "schema": "fs-traces-packed-1",
                    "path": path.relative_to(result).as_posix(),
                    "retention": "durable",
                    "bytes": path.stat().st_size,
                    "generation": f"task-{task}",
                    "dataset_number": 7 + task,
                }
            )
        record = result / f"_fs_run/tasks/task_{task:06d}/result.json"
        record.parent.mkdir(parents=True)
        record.write_text(
            json.dumps(
                {
                    "schema": "fs-task-result-2",
                    "partition": {
                        "task": task,
                        "frequency": {"real": frequency, "imag": 0.0},
                    },
                    "status": {"state": "success", "code": 0},
                    "fingerprints": {
                        key: "a" * 64 for key in ("job", "simulation", "outputs")
                    },
                    "artifacts": artifacts,
                }
            )
        )
    products = TraceManifest._sampled_wavefield_packs(result, (1.0, 2.0))
    assert len(products) == 2
    manifest = TraceManifest(
        files=[segment.path for product in products for segment in product.segments],
        frequencies={1: 1.0, 2: 2.0},
        groups=["surface", "other"],
        simulation=tmp_path / "simulation.json",
        result_path=result,
        output_path=result / "fields",
        wavefield_packs=products,
    )
    with TraceDataset.from_manifest(manifest) as traces:
        assert set(traces.groups) == {"surface", "other"}
        for offset, group in enumerate(("surface", "other")):
            gather = traces.fd(group, "p", source=1).compute()
            assert gather.coords["frequency"].values.tolist() == [1.0, 2.0]
            assert gather.real.values.tolist() == [
                [1.0 + offset * 10] * 2,
                [2.0 + offset * 10] * 2,
            ]
