import numpy as np

import frequensolve as fs


def _receiver_device():
    device = fs.ReceiverNode(name="node")
    device.add_component(name="pressure", field="pressure")
    return device


def test_point_sources_export_v2_source_geometry():
    acq = fs.Acquisition(
        sources=fs.SourceGeometry.points(
            kind="scalar",
            coords=[[0.25, 0.05], [0.5, 0.05]],
            names=["s001", "s002"],
            units="km",
            domain=0,
            mechanism="isotropic",
        ),
        max_batch=4,
    )

    payload = acq.to_fs()

    assert payload["schema"] == "fs-acquisition-2"
    assert payload["max_batch"] == 4
    assert payload["source_geometry"] == {
        "_type": "Inline",
        "domain": 0,
        "kind": "scalar",
        "defaults": {"mechanism": {"type": "isotropic"}},
        "sources": [
            {
                "name": "s001",
                "coordinates": {"value": [0.25, 0.05], "units": "km"},
            },
            {
                "name": "s002",
                "coordinates": {"value": [0.5, 0.05], "units": "km"},
            },
        ],
    }
    assert "source_encoding" not in payload
    assert acq.source_field_ids() == [1, 2]
    assert acq.source_field_names() == ["s001", "s002"]


def test_named_source_encoding_exports_sparse_complex_terms():
    acq = fs.Acquisition(
        sources=fs.SourceGeometry.inline(
            kind="vector",
            sources=[
                fs.PointSource("s001", fs.CoordinateValue([0.0, 0.1], units="km")),
                fs.PointSource("s002", fs.CoordinateValue([0.1, 0.1], units="km")),
            ],
            defaults={"direction": [0.0, 1.0]},
        ),
        source_encoding=fs.SourceEncoding.named(
            [fs.DistributedSource("distributed_001", {"s001": 1.0, "s002": 1j})]
        ),
        max_batch=1,
    )

    payload = acq.to_fs()

    assert payload["source_encoding"] == {
        "_type": "Named",
        "fields": [
            {
                "name": "distributed_001",
                "terms": [
                    {"source": "s001", "coefficient": 1.0},
                    {"source": "s002", "coefficient": [0.0, 1.0]},
                ],
            }
        ],
    }
    assert acq.source_field_ids() == [1]
    assert acq.source_field_names() == ["distributed_001"]
    assert acq.source_coords().tolist() == [[0.05, 0.1]]


def test_dense_source_encoding_accepts_source_major_matrix():
    encoding = fs.SourceEncoding.dense(
        np.asarray([[1.0, 0.0], [0.0, 1j]]),
        names=["mode_1", "mode_2"],
        reference_coordinates=[[0.0, 0.1], [0.1, 0.1]],
    )

    assert encoding.to_fs() == {
        "_type": "JsonDense",
        "fields": [
            {
                "name": "mode_1",
                "coefficients": [1.0, 0.0],
                "reference_coordinates": [0.0, 0.1],
            },
            {
                "name": "mode_2",
                "coefficients": [0.0, [0.0, 1.0]],
                "reference_coordinates": [0.1, 0.1],
            },
        ],
    }


def test_acquisition_add_distributed_source_appends_named_field():
    acq = fs.Acquisition()
    acq.add_sources(
        kind="scalar",
        coords=[[0.0, 0.0], [1.0, 0.0]],
        names=["left", "right"],
    )
    acq.add_distributed_source("dipole_like", {"left": 1.0, "right": -1.0})

    assert acq.to_fs()["source_encoding"]["fields"][0]["name"] == "dipole_like"
    assert acq.source_coords().tolist() == [[0.5, 0.0]]


def test_survey_to_acquisition_uses_v2_source_geometry():
    survey = fs.Survey.dense(
        "line",
        sources=[[0.0, 0.0], [1.0, 0.0]],
        receivers=[[0.0, 1.0], [1.0, 1.0]],
    )

    payload = survey.to_acquisition(_receiver_device()).to_fs()

    assert payload["schema"] == "fs-acquisition-2"
    assert payload["source_geometry"]["_type"] == "Inline"
    assert len(payload["source_geometry"]["sources"]) == 2
    assert len(payload["receiver_groups"]) == 1
    assert "source_groups" not in payload


def test_acquisition_constructor_accepts_receivers_alias():
    receivers = [
        fs.ReceiverGroup(
            "surface",
            _receiver_device(),
            [[0.0, 0.0], [1.0, 0.0]],
        )
    ]
    acq = fs.Acquisition(
        sources=fs.SourceGeometry.points(kind="scalar", coords=[[0.5, 0.0]]),
        receivers=receivers,
    )

    assert acq.receivers is acq.receiver_groups
    assert acq.to_fs()["receiver_groups"][0]["name"] == "surface"
