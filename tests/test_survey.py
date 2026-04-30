import numpy as np
import pytest

from frequensolve.seismic import ReceiverNode, Survey
from frequensolve.seismic.sparse_survey import SparseSurvey


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


def test_dense_survey_populates_dense_acquisition():
    survey = Survey.dense(
        "line",
        sources=[[0.0, 0.0], [1.0, 0.0]],
        receivers=[[0.0, 1.0], [1.0, 1.0]],
    )

    payload = survey.to_acquisition(_device()).to_fs()

    assert len(payload["source_groups"]) == 2
    assert len(payload["receiver_groups"]) == 1
    assert "frame" not in payload["source_groups"][0]["source"]
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

    assert payload["source_groups"][0]["source"]["coordinates"] == {
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
