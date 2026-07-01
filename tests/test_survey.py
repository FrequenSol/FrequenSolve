from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from frequensolve.orchestrator.sites.base import JobStatus, RunResult
from frequensolve.seismic import ReceiverNode, Survey
from frequensolve.seismic.sparse_survey import SparseSurvey
from frequensolve.simulation.jobs.artifacts import RunMetadata


def _field(line, i1, i2, value):
    text = str(value)
    width = i2 - i1 + 1
    line[i1 - 1 : i2] = f"{text:>{width}}"[-width:]


def _point(kind, line_number, point, index, x, z):
    line = [" "] * 80
    line[0] = kind
    _field(line, 2, 17, f"{line_number:.1f}")
    _field(line, 18, 25, f"{point:.1f}")
    _field(line, 26, 26, index)
    _field(line, 47, 55, f"{x:.1f}")
    _field(line, 56, 65, f"{z:.1f}")
    return "".join(line) + "\n"


def _relation(
    field_record,
    source_line,
    source_point,
    from_channel,
    to_channel,
    receiver_line,
    from_receiver,
    to_receiver,
):
    line = [" "] * 80
    line[0] = "X"
    _field(line, 8, 15, field_record)
    _field(line, 18, 27, f"{source_line:.1f}")
    _field(line, 28, 37, f"{source_point:.1f}")
    _field(line, 38, 38, 1)
    _field(line, 39, 43, from_channel)
    _field(line, 44, 48, to_channel)
    _field(line, 49, 49, 1)
    _field(line, 50, 59, f"{receiver_line:.1f}")
    _field(line, 60, 69, f"{from_receiver:.1f}")
    _field(line, 70, 79, f"{to_receiver:.1f}")
    _field(line, 80, 80, 1)
    return "".join(line) + "\n"


def _device():
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    return hydrophone


def _write_h5_strings(group, name, values):
    group.create_dataset(
        name,
        data=np.asarray(values, dtype=object),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _set_h5_string_attr(item, name, values):
    item.attrs.create(
        name,
        np.asarray(values, dtype=object),
        dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _write_solver_catalog(
    survey_group,
    catalog_name,
    *,
    id_name,
    name_name,
    ids,
    names,
    coordinates,
    coordinate_attrs=True,
):
    catalog = survey_group.create_group(catalog_name)
    catalog.create_dataset(id_name, data=np.asarray(ids, dtype=np.int64))
    _write_h5_strings(catalog, name_name, names)
    coords = catalog.create_dataset(
        "coordinates", data=np.asarray(coordinates, dtype=float)
    )
    if coordinate_attrs:
        _set_h5_string_attr(coords, "units", ["km"])
        _set_h5_string_attr(coords, "coordinate_system", ["model"])
    return catalog


def _write_solver_trace_store(
    path, *, layout="sparse", group_name="surface", dataset_name=None
):
    dataset_name = dataset_name or group_name
    with h5py.File(path, "w") as h5:
        h5.create_dataset("frequency", data=10.0)
        survey_group = h5.create_group("survey")
        _write_h5_strings(survey_group, "schema_version", ["fs_seismic_trace_store_v1"])
        _write_h5_strings(survey_group, "layout_kind", [f"{layout}_trace_v1"])
        _write_solver_catalog(
            survey_group,
            "sources",
            id_name="source_id",
            name_name="source_name",
            ids=[7, 8],
            names=["source_7", "source_8"],
            coordinates=[[0.0, 0.0], [1.0, 0.0]],
        )
        survey_group["sources"].create_dataset(
            "field_record", data=np.asarray([700, 800], dtype=np.int64)
        )
        components = survey_group.create_group("components")
        components.create_dataset(
            "component_id", data=np.asarray([1, 2], dtype=np.int64)
        )
        components.create_dataset("component", data=np.asarray([1, 2], dtype=np.int64))
        _write_h5_strings(components, "component_name", ["p", "vx"])
        receiver_groups = survey_group.create_group("receiver_groups")
        catalog = receiver_groups.create_group("_catalog")
        _write_h5_strings(catalog, "group_name", [group_name])
        _write_h5_strings(catalog, "dataset_path", [f"/{dataset_name}"])
        _write_h5_strings(catalog, "layout_kind", [f"{layout}_trace_v1"])
        catalog.create_dataset("receiver_count", data=np.asarray([3], dtype=np.int64))
        catalog.create_dataset("component_count", data=np.asarray([2], dtype=np.int64))
        catalog.create_dataset("source_count", data=np.asarray([2], dtype=np.int64))

        if layout == "sparse":
            _write_solver_catalog(
                survey_group,
                "receivers",
                id_name="receiver_id",
                name_name="receiver_name",
                ids=[101, 102, 103],
                names=["receiver_101", "receiver_102", "receiver_103"],
                coordinates=[[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
                coordinate_attrs=False,
            )
            traces = survey_group.create_group(f"receiver_groups/{dataset_name}/traces")
            traces.create_dataset(
                "trace_id", data=np.asarray([1, 2, 3, 4], dtype=np.int64)
            )
            traces.create_dataset(
                "source_id", data=np.asarray([7, 7, 8, 8], dtype=np.int64)
            )
            traces.create_dataset(
                "receiver_id", data=np.asarray([101, 102, 102, 103], dtype=np.int64)
            )
            traces.create_dataset(
                "component", data=np.asarray([1, 1, 2, 1], dtype=np.int64)
            )
            traces.create_dataset(
                "weight", data=np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
            )
        elif layout == "dense":
            _write_solver_catalog(
                survey_group,
                f"receiver_groups/{dataset_name}/receivers",
                id_name="receiver_id",
                name_name="receiver_name",
                ids=[1, 2, 3],
                names=[f"{group_name}_1", f"{group_name}_2", f"{group_name}_3"],
                coordinates=[[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            )
            group_components = survey_group.create_group(
                f"receiver_groups/{dataset_name}/components"
            )
            group_components.create_dataset(
                "component_id", data=np.asarray([1, 2], dtype=np.int64)
            )
            group_components.create_dataset(
                "component", data=np.asarray([1, 2], dtype=np.int64)
            )
            _write_h5_strings(group_components, "component_name", ["p", "vx"])
            data = h5.create_dataset(
                dataset_name, data=np.zeros((3, 2, 2), dtype=np.float32)
            )
            _set_h5_string_attr(data, "layout_kind", ["dense_trace_v1"])
            _set_h5_string_attr(data, "dims", ["receiver", "component", "shot"])
            data.attrs["receiver"] = np.asarray([1, 2, 3], dtype=np.int64)
            data.attrs["shot"] = np.asarray([7, 8], dtype=np.int64)
            _set_h5_string_attr(data, "component", ["p", "vx"])
        else:
            raise ValueError(layout)


def _write_flat_sparse_trace_store(path, *, aligned=False):
    with h5py.File(path, "w") as h5:
        h5.create_dataset("frequency", data=10.0)
        survey_group = h5.create_group("survey")
        _write_h5_strings(survey_group, "schema_version", ["fs_seismic_trace_store_v1"])
        _write_h5_strings(survey_group, "layout_kind", ["sparse_trace_v1"])
        _write_solver_catalog(
            survey_group,
            "sources",
            id_name="source_id",
            name_name="source_name",
            ids=[7, 8],
            names=["source_7", "source_8"],
            coordinates=[[0.0, 0.0], [1.0, 0.0]],
        )
        survey_group["sources"].create_dataset(
            "field_record", data=np.asarray([700, 800], dtype=np.int64)
        )
        _write_solver_catalog(
            survey_group,
            "receivers",
            id_name="receiver_id",
            name_name="receiver_name",
            ids=[101, 102],
            names=["receiver_101", "receiver_102"],
            coordinates=[[0.0, 1.0], [1.0, 1.0]],
            coordinate_attrs=False,
        )
        components = survey_group.create_group("components")
        components.create_dataset(
            "component_id", data=np.asarray([11, 12], dtype=np.int64)
        )
        components.create_dataset("component", data=np.asarray([1, 2], dtype=np.int64))
        _write_h5_strings(components, "component_name", ["p", "vx"])

        traces = survey_group.create_group("traces")
        if aligned:
            _write_h5_strings(traces, "layout_encoding", ["aligned_components_v1"])
            nodes = survey_group.create_group("trace_nodes")
            nodes.create_dataset("source_id", data=np.asarray([7, 8], dtype=np.int64))
            nodes.create_dataset(
                "receiver_id", data=np.asarray([101, 102], dtype=np.int64)
            )
            nodes.create_dataset(
                "weight", data=np.asarray([1.0, 0.0], dtype=np.float32)
            )
        else:
            traces.create_dataset(
                "trace_id", data=np.asarray([1, 2, 3], dtype=np.int64)
            )
            traces.create_dataset(
                "source_id", data=np.asarray([7, 7, 8], dtype=np.int64)
            )
            traces.create_dataset(
                "receiver_id", data=np.asarray([101, 102, 101], dtype=np.int64)
            )
            traces.create_dataset(
                "component", data=np.asarray([1, 2, 1], dtype=np.int64)
            )
            traces.create_dataset(
                "weight", data=np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
            )


def test_dense_survey_populates_dense_acquisition():
    survey = Survey.dense(
        "line",
        sources=[[0.0, 0.0], [1.0, 0.0]],
        receivers=[[0.0, 1.0], [1.0, 1.0]],
    )

    payload = survey.to_acquisition(_device()).to_fs()

    assert len(payload["source_geometry"]["sources"]) == 2
    assert len(payload["receiver_groups"]) == 1
    assert "frame" not in payload["source_geometry"]["sources"][0]
    assert "frame" not in payload["receiver_groups"][0]
    assert "sampling" not in payload["receiver_groups"][0]
    assert "surveys" not in payload

    with pytest.raises(TypeError, match="frame"):
        survey.to_acquisition(_device(), source_frame="reference")


def test_dense_survey_preserves_units_and_coordinate_systems():
    survey = Survey.dense(
        "surface_line",
        sources=[[0.5, 0.0]],
        receivers=[[0.0, 0.0], [1.0, 0.0]],
        units="km",
        source_system="source_depth",
        receiver_system="free_surface",
    )

    payload = survey.to_acquisition(_device()).to_fs()

    assert payload["source_geometry"]["sources"][0]["coordinates"] == {
        "value": [0.5, 0.0],
        "units": "km",
        "system": "source_depth",
    }
    assert payload["receiver_groups"][0]["coordinates"] == {
        "_type": "CoordsArray",
        "value": [[0.0, 0.0], [1.0, 0.0]],
        "units": "km",
        "system": "free_surface",
    }


def test_offset_domain_survey_exports_compact_solver_rule():
    survey = Survey.offset_domain(
        "near_offsets",
        sources=[[0.0, 0.0]],
        receivers=[[0.25, 0.0], [2.0, 0.0]],
        min=0.0,
        max=1.0,
        metric="horizontal",
    )

    payload = survey.to_acquisition(_device()).to_fs()

    assert payload["receiver_groups"][0]["sampling"] == {
        "_type": "OffsetDomain",
        "survey": "near_offsets",
    }
    assert payload["surveys"][0] == {
        "name": "near_offsets",
        "_type": "OffsetDomain",
        "offset_domain": {
            "metric": "horizontal",
            "absolute": True,
            "min": 0.0,
            "max": 1.0,
        },
    }
    assert SparseSurvey.from_fs(payload["surveys"][0]).to_fs() == payload["surveys"][0]


def test_sps_survey_parses_sources_receivers_and_spx_relations(tmp_path):
    sps = tmp_path / "line.sps"
    spr = tmp_path / "line.spr"
    spx = tmp_path / "line.spx"
    sps.write_text(_point("S", 10, 1, 1, 0.0, 0.0))
    spr.write_text(_point("R", 20, 1, 1, 0.0, 1.0) + _point("R", 20, 2, 1, 1.0, 1.0))
    spx.write_text(_relation(101, 10, 1, 1, 2, 20, 1, 2))

    survey = Survey.from_sps(sps, spr, spx, name="sps_line")

    assert survey.source_ids.tolist() == [1]
    assert survey.receiver_ids.tolist() == [1, 2]
    assert survey.sources.tolist() == [[0.0, 0.0]]
    assert survey.receivers.tolist() == [[0.0, 1.0], [1.0, 1.0]]
    assert survey.relations["source_id"].tolist() == [1, 1]
    assert survey.relations["receiver_id"].tolist() == [1, 2]
    assert survey.relations["channel_number"].tolist() == [1, 2]

    payload = survey.to_acquisition(_device()).to_fs()
    assert payload["receiver_groups"][0]["sampling"] == {
        "_type": "SPSFiles",
        "survey": "sps_line",
    }
    assert payload["surveys"][0]["source_file"] == str(sps)
    assert payload["surveys"][0]["receiver_file"] == str(spr)
    assert payload["surveys"][0]["relation_file"] == str(spx)


def test_survey_plot_filters_corresponding_receivers(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sps = tmp_path / "line.sps"
    spr = tmp_path / "line.spr"
    spx = tmp_path / "line.spx"
    sps.write_text(_point("S", 10, 1, 1, 0.0, 0.0))
    spr.write_text(_point("R", 20, 1, 1, 0.0, 1.0) + _point("R", 20, 2, 1, 1.0, 1.0))
    spx.write_text(_relation(101, 10, 1, 1, 2, 20, 1, 2))
    survey = Survey.from_sps(sps, spr, spx, name="sps_line")

    ax = survey.plot(sources=1)

    assert len(ax.lines) == 2
    assert np.asarray(ax.collections[0].get_offsets()).shape == (1, 2)
    assert np.asarray(ax.collections[1].get_offsets()).shape == (2, 2)
    plt.close(ax.figure)


def test_survey_loads_sparse_solver_trace_store(tmp_path):
    trace_file = tmp_path / "traces_00001.h5"
    _write_solver_trace_store(trace_file, layout="sparse")

    survey = Survey.from_trace_file(trace_file, group="surface", name="loaded")

    assert survey.name == "loaded"
    assert survey.kind == "SolverTraceStore"
    assert survey.source_ids.tolist() == [7, 8]
    assert survey.receiver_ids.tolist() == [101, 102, 103]
    assert survey.sources.tolist() == [[0.0, 0.0], [1.0, 0.0]]
    assert survey.receivers.tolist() == [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    assert survey.source_units == "km"
    assert survey.receiver_system is None
    assert list(survey.trace_tables) == ["surface"]
    assert survey.trace_tables["surface"]["active"].tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert survey.trace_tables["surface"]["weight"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert survey.trace_tables["surface"]["component_id"].tolist() == [1, 1, 2, 1]
    assert survey.relations["source_id"].tolist() == [7, 7, 8]
    assert survey.relations["receiver_id"].tolist() == [101, 102, 102]
    assert survey.relations["channel_number"].tolist() == [1, 2, 3]
    assert survey.relations["field_record"].tolist() == [700, 700, 800]


def test_survey_loads_flat_sparse_solver_trace_store(tmp_path):
    trace_file = tmp_path / "store.h5"
    _write_flat_sparse_trace_store(trace_file)

    survey = Survey.from_trace_file(trace_file)

    assert list(survey.trace_tables) == ["traces"]
    table = survey.trace_tables["traces"]
    assert table["active"].tolist() == [True, False, True]
    assert table["component_id"].tolist() == [11, 12, 11]
    assert table["field_record"].tolist() == [700, 700, 800]
    assert survey.relations["source_id"].tolist() == [7, 8]
    assert survey.relations["receiver_id"].tolist() == [101, 101]


def test_survey_loads_aligned_component_sparse_solver_trace_store(tmp_path):
    trace_file = tmp_path / "store.h5"
    _write_flat_sparse_trace_store(trace_file, aligned=True)

    survey = Survey.from_trace_file(trace_file)

    table = survey.trace_tables["traces"]
    assert table["trace_id"].tolist() == [1, 2, 3, 4]
    assert table["source_id"].tolist() == [7, 7, 8, 8]
    assert table["receiver_id"].tolist() == [101, 101, 102, 102]
    assert table["component_id"].tolist() == [11, 12, 11, 12]
    assert table["component"].tolist() == [1, 2, 1, 2]
    assert table["active"].tolist() == [True, True, False, False]
    assert survey.relations["source_id"].tolist() == [7]
    assert survey.relations["receiver_id"].tolist() == [101]


def test_survey_loads_dense_solver_trace_store_without_receiver_group_table(tmp_path):
    trace_file = tmp_path / "traces_00001.h5"
    _write_solver_trace_store(trace_file, layout="dense")

    survey = Survey.from_trace_file(trace_file)

    assert list(survey.trace_tables) == ["surface"]
    table = survey.trace_tables["surface"]
    assert len(table) == 12
    assert table["source_id"].tolist()[:6] == [7, 7, 7, 7, 7, 7]
    assert table["receiver_id"].tolist()[:4] == [1, 1, 2, 2]
    assert table["component_id"].tolist()[:4] == [1, 2, 1, 2]
    assert survey.relations["source_id"].tolist() == [7, 7, 7, 8, 8, 8]
    assert survey.relations["receiver_id"].tolist() == [1, 2, 3, 1, 2, 3]


def test_survey_loads_group_by_catalog_name_when_dataset_path_differs(tmp_path):
    trace_file = tmp_path / "traces_00001.h5"
    _write_solver_trace_store(
        trace_file,
        layout="dense",
        group_name="surface",
        dataset_name="surface_inc",
    )

    survey = Survey.from_trace_file(trace_file, group="surface")

    assert list(survey.trace_tables) == ["surface"]
    assert survey.receiver_names.tolist() == ["surface_1", "surface_2", "surface_3"]


def test_survey_loads_trace_store_from_run_result_metadata(tmp_path):
    trace_file = tmp_path / "traces_00001.h5"
    _write_solver_trace_store(trace_file, layout="sparse")
    result = RunResult(
        job=None,
        status=JobStatus(state="completed", return_code=0),
        run_metadata=RunMetadata(
            outputs={
                "files": [
                    {
                        "relative_path": trace_file.name,
                        "kind": "hdf5",
                        "schema": "fs_seismic_trace_store_v1",
                    }
                ]
            },
            result_path=tmp_path,
        ),
    )

    survey = Survey.from_result(result)

    assert survey.source_ids.tolist() == [7, 8]
    assert survey.receiver_ids.tolist() == [101, 102, 103]


def test_survey_load_dispatches_supported_inputs(tmp_path):
    trace_file = tmp_path / "traces_00001.h5"
    _write_solver_trace_store(trace_file, layout="sparse")
    result = RunResult(
        job=None,
        status=JobStatus(state="completed", return_code=0),
        run_metadata=RunMetadata(
            outputs={
                "files": [
                    {
                        "relative_path": trace_file.name,
                        "kind": "hdf5",
                        "schema": "fs_seismic_trace_store_v1",
                    }
                ]
            },
            result_path=tmp_path,
        ),
    )

    assert Survey.load(trace_file).receiver_ids.tolist() == [101, 102, 103]
    assert Survey.load(result).source_ids.tolist() == [7, 8]
    assert Survey.load(SimpleNamespace(paths=[trace_file])).kind == "SolverTraceStore"

    sps = tmp_path / "line.sps"
    spr = tmp_path / "line.spr"
    spx = tmp_path / "line.spx"
    sps.write_text(_point("S", 10, 1, 1, 0.0, 0.0))
    spr.write_text(_point("R", 20, 1, 1, 0.0, 1.0))
    spx.write_text(_relation(101, 10, 1, 1, 1, 20, 1, 1))

    assert Survey.load((sps, spr, spx)).kind == "SPSFiles"
    assert Survey.load(sps, spr, spx).receiver_ids.tolist() == [1]
