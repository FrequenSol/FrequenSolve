"""Scientific plot labels preserve coordinates, finite values and principal phase."""

from copy import deepcopy

import numpy as np
import pytest

from frequensolve.workflow.analysis import analyze_pressure
from frequensolve.workflow.presentation import (
    figure_png,
    geometry,
    geometry_labels,
    present,
)
from tests.test_workflow import snapshot
from tests.test_workflow_presentation import view


@pytest.fixture
def rendered(monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    saved, original = [], Figure.savefig

    def capture(figure, *args, **kwargs):
        saved.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return saved


def test_labels_combine_only_exactly_colocated_input_points():
    model = {
        "sources": np.array([[0.25, 0.5, 0.2], [0.75, 0.5, 0.2]]),
        "receivers": np.array(
            [[0.75, 0.5, 0.2], [0.25, 0.5, 0.2], [0.250000001, 0.5, 0.2]]
        ),
    }
    assert geometry_labels(model) == [
        {"point": (0.25, 0.5, 0.2), "label": "S1 / R2"},
        {"point": (0.75, 0.5, 0.2), "label": "S2 / R1"},
        {"point": (0.250000001, 0.5, 0.2), "label": "R3"},
    ]


def test_model_figure_and_table_label_sources_and_receivers_in_input_order(rendered):
    value = view()
    acquisition = value["plan"]["scientific_inputs"]["acquisition"]
    acquisition["source_geometry"]["sources"].append({"coordinates": [0.75, 0.5, 0.2]})
    acquisition["receiver_groups"][0]["coordinates"]["value"] = [
        [0.25, 0.5, 0.2],
        [0.75, 0.5, 0.2],
    ]
    before = deepcopy(value)
    output = present(value, "model", [0, 1])
    assert value == before
    assert output.provenance["geometry_label_origin"] == "input_order"
    assert all(
        label in output.narrative
        for label in (r"S1 \(source\)", r"R1 \(receiver\)", r"R2 \(receiver\)")
    )
    assert "not a simulated wavefield" in output.narrative
    axes = rendered[0].axes[0]
    assert {t.get_text().strip() for t in axes.texts} == {"S1 / R1", "S2 / R2"}
    # Physical box proportions and displayed units remain unchanged.
    aspect = axes.get_box_aspect()
    np.testing.assert_allclose(aspect / aspect[0], [1, 2, 1])
    assert axes.get_xlabel() == "X (km)" and axes.get_zlabel() == "Z (km)"
    assert output.figures[0].startswith(b"\x89PNG") and len(output.figures[0]) < 1048576


def test_many_geometry_points_keep_labels_in_the_complete_coordinate_table(rendered):
    value = view()
    acquisition = value["plan"]["scientific_inputs"]["acquisition"]
    acquisition["source_geometry"]["sources"] = [
        {"coordinates": [i / 32, 0.5, 0.2]} for i in range(16)
    ]
    acquisition["receiver_groups"][0]["coordinates"]["value"] = [
        [i / 32, 0.6, 0.2] for i in range(16)
    ]
    model = geometry(value["plan"]["scientific_inputs"])
    assert geometry_labels(model) == []
    output = present(value, "model", [0, 1])
    assert (
        r"S16 \(source\)" in output.narrative
        and r"R16 \(receiver\)" in output.narrative
    )
    assert [t.get_text() for t in rendered[0].axes[0].texts] == [
        "Point labels are listed in the coordinate table"
    ]
    assert len(output.figures[0]) < 1048576


def test_small_nonzero_zero_and_missing_pressures_have_truthful_rendered_marks(
    rendered,
):
    data = snapshot([3 + 4j, 0, 0.03 + 0.04j, complex(float("nan"), 0)])
    report = analyze_pressure(data)
    before = deepcopy(report)
    png = figure_png(reports=[report])
    assert report == before
    magnitude, phase = rendered[0].axes
    values = magnitude.collections[0].get_offsets()
    np.testing.assert_allclose(values[:3, 1], [5, 0, 0.05])
    assert np.ma.getmaskarray(values)[3].all()
    assert {t.get_text() for t in magnitude.texts} == {
        "5",
        "0",
        "0.05",
        "Undefined values omitted",
    }
    assert magnitude.get_ylabel() == "Pa" and magnitude.get_ylim()[0] == 0
    assert [t.get_text() for t in magnitude.get_xticklabels()] == [
        "S1→R0",
        "S1→R1",
        "S1→R2",
        "S1→R3",
    ]
    phase_values = phase.collections[0].get_offsets()
    assert np.ma.getmaskarray(phase_values)[1].all()  # Zero magnitude has no phase.
    assert np.ma.getmaskarray(phase_values)[3].all()
    assert phase.get_ylabel() == "rad"
    np.testing.assert_allclose(phase.get_yticks(), [-np.pi, 0, np.pi])
    assert png.startswith(b"\x89PNG") and len(png) < 1048576


def test_principal_phase_wraps_without_unwrapping_and_larger_selections_avoid_labels(
    rendered,
):
    report = analyze_pressure(snapshot([-1 + 1e-12j, -1 - 1e-12j]))
    figure_png(reports=[report])
    phase = rendered[0].axes[1].collections[0].get_offsets()[:, 1]
    assert phase[0] > 3.14 and phase[1] < -3.14
    report = analyze_pressure(snapshot([1j] * 16))
    png = figure_png(reports=[report])
    assert not rendered[1].axes[0].texts
    np.testing.assert_allclose(
        rendered[1].axes[0].collections[0].get_offsets()[:, 1], [1] * 16
    )
    assert len(png) < 1048576


def test_pressure_plot_annotations_leave_scientific_provenance_unchanged(rendered):
    value = view()
    before = deepcopy(value)
    with_figure = present(value, "results", [0, 1])
    without_figure = present(value, "results", [0, 1], False)
    assert with_figure.provenance == without_figure.provenance
    assert value == before
