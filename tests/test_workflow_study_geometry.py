"""Native geometry labels preserve recorded identities and coordinate positions."""

from copy import deepcopy

import pytest

from frequensolve.workflow.study import figure_svg, study_report
from tests.test_workflow_study import data


def plot(monkeypatch, metadata):
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = []
    original_close = plt.close

    def close(fig=None):
        if hasattr(fig, "savefig"):
            captured.append(fig)
        original_close(fig)

    monkeypatch.setattr(plt, "close", close)
    svg = figure_svg([{"metadata": metadata}], "geometry")
    assert svg.startswith("<svg")
    return captured[-1].axes[0]


def native_points():
    return {
        "source_ids": [7, 19],
        "receiver_ids": [42, 99],
        "source_coordinates": [[250.0, 500.0, 200.0], [750.0, 500.0, 200.0]],
        "receiver_coordinates": [[250.0, 500.0, 200.0], [750.0, 500.0, 200.0]],
        "coordinate_units": "m",
    }


def point_texts(axes):
    return {
        text.get_text().strip(): tuple(text.get_position_3d())
        for text in axes.texts
        if hasattr(text, "get_position_3d")
    }


def test_exact_colocation_groups_native_labels_without_renumbering_or_moving(
    monkeypatch,
):
    metadata = native_points()
    before = deepcopy(metadata)
    axes = plot(monkeypatch, metadata)
    assert point_texts(axes) == {
        "S7 / R42": (250.0, 500.0, 200.0),
        "S19 / R99": (750.0, 500.0, 200.0),
    }
    assert [axes.get_xlabel(), axes.get_ylabel(), axes.get_zlabel()] == [
        "X (m)",
        "Y (m)",
        "Z (m)",
    ]
    assert len(axes.collections) == 2
    source, receiver = axes.collections
    assert len(source.get_facecolors()) > 0
    assert len(receiver.get_facecolors()) == 0 and len(receiver.get_edgecolors()) > 0
    assert metadata == before


def test_near_coordinates_remain_distinct_and_unshifted(monkeypatch):
    metadata = native_points()
    metadata["receiver_coordinates"][0][0] += 1e-8
    axes = plot(monkeypatch, metadata)
    labels = point_texts(axes)
    assert set(labels) == {"S7", "R42", "S19 / R99"}
    assert labels["S7"] == tuple(metadata["source_coordinates"][0])
    assert labels["R42"] == tuple(metadata["receiver_coordinates"][0])
    assert labels["S7"] != labels["R42"]


def test_dense_geometry_keeps_points_and_explicitly_bounds_label_clutter(monkeypatch):
    coordinates = [[float(i), 0.0, 0.0] for i in range(7)]
    metadata = {
        "source_ids": list(range(10, 17)),
        "receiver_ids": list(range(40, 47)),
        "source_coordinates": coordinates,
        "receiver_coordinates": deepcopy(coordinates),
        "coordinate_units": "km",
    }
    before = deepcopy(metadata)
    axes = plot(monkeypatch, metadata)
    assert not point_texts(axes)
    assert any(
        "Point labels omitted for dense geometry" in text.get_text()
        for text in axes.texts
    )
    assert [len(collection.get_offsets()) for collection in axes.collections] == [7, 7]
    assert axes.get_xlabel() == "X (km)" and metadata == before


def test_native_geometry_does_not_modify_analysis_arrays_or_provenance():
    pytest.importorskip("matplotlib")
    context, snapshots, references = data()
    report = study_report(context, snapshots, references)
    before = deepcopy(report)
    figure_svg(report["reports"], "geometry")
    assert report == before
    assert report["reports"][0]["magnitude"] == [[5, 0]]
    assert report["reports"][0]["principal_phase_radians"][0][1] is None
