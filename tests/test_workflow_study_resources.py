"""Customers can read saved choices without inferring capacity or execution."""

import json
from copy import deepcopy
from html.parser import HTMLParser

import pytest

from frequensolve.workflow.study import (
    create_study_notebook,
    report_html,
    resource_report_html,
    study_report,
)
from tests.test_workflow_study import data


class ResourceView(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.rows, self.headers, self.words, self.tags, self.attributes = (
            [],
            [],
            [],
            [],
            [],
        )
        self.row, self.cell, self.kind, self.details = [], None, None, 0
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend(attrs)
        if tag == "details":
            self.details += 1
        if tag == "tr":
            self.row = []
        if tag in ("td", "th"):
            self.cell, self.kind = [], tag

    def handle_data(self, value):
        if not self.details:
            self.words.append(value)
        if self.cell is not None:
            self.cell.append(value)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            value = "".join(self.cell)
            (self.headers if tag == "th" else self.row).append(value)
            self.cell, self.kind = None, None
        if tag == "tr" and self.row:
            self.rows.append(self.row)
        if tag == "details":
            self.details -= 1

    @property
    def text(self):
        return " ".join(" ".join(self.words).split())


def plan_view():
    rows = []
    for frequency, nodes, memory in [(2, 1, 1728), (20, 2, 73365)]:
        automatic = {
            "layout": {
                "node_type": "example-cpu",
                "partition": "cpu",
                "nodes": nodes,
                "ranks_per_node": 1,
                "cpus_per_rank": 8,
            },
            "memory_mib_per_node": memory,
            "wall_time_seconds": 1800,
        }
        requested = deepcopy(automatic)
        if frequency == 2:
            requested["memory_mib_per_node"] = 2240
        rows.append(
            {
                "frequency": str(frequency),
                "automatic": automatic,
                "requested": requested,
                "estimated_runtime_seconds": None,
                "validation": {"status": "valid"},
                "reason": "Fewest feasible nodes within the qualified CPU profile",
            }
        )
    return {
        "plan": {
            "version": "frequency-execution-plan.v1",
            "revision": 3,
            "rows": rows,
            "limits": {
                "max_concurrent_frequencies": 2,
                "max_nodes_per_frequency": 2,
                "max_running_cpus": 64,
            },
        }
    }


def test_readable_per_frequency_choices_and_qualified_automatic_difference():
    value = plan_view()
    before = deepcopy(value)
    view = ResourceView(resource_report_html(value))
    assert view.rows == [
        [
            "2 Hz",
            "1",
            "example-cpu",
            "cpu",
            "1",
            "8",
            "2240",
            "1800",
            "Unavailable",
            "valid (recorded)",
        ],
        [
            "20 Hz",
            "2",
            "example-cpu",
            "cpu",
            "1",
            "8",
            "73365",
            "1800",
            "Unavailable",
            "valid (recorded)",
        ],
    ]
    assert view.headers == [
        "Frequency",
        "Nodes",
        "Node type",
        "Partition",
        "Ranks per node",
        "CPUs per rank",
        "Memory per node (MiB)",
        "Wall-time limit (s)",
        "Estimated runtime (s)",
        "Validation",
    ]
    assert "Memory per node: 1728 → 2240 MiB" in view.text
    assert (
        "20 Hz: Requested resources match the saved automatic suggestion." in view.text
    )
    assert "Memory per node is a request, not measured peak memory" in view.text
    assert "Wall time is a limit, not a runtime estimate" in view.text
    assert "no new validation or authorization" in view.text
    assert (
        "concurrent frequencies: 2; nodes per frequency: 2; running CPUs: 64"
        in view.text
    )
    assert "do not report available cluster capacity" in view.text
    assert value == before


@pytest.mark.parametrize(
    "bad", [None, False, "NaN", "Infinity", -1, 0, 1.5, "1e100000"]
)
def test_invalid_requested_counts_stay_unavailable_without_automatic_fallback(bad):
    value = plan_view()
    value["plan"]["rows"][0]["requested"]["layout"]["nodes"] = bad
    view = ResourceView(resource_report_html(value))
    assert view.rows[0][1] == "Unavailable"
    assert "Other fields could not be compared" in view.text


@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"plan": None},
        {"plan": {"version": "future"}},
        {"plan": {"version": "frequency-execution-plan.v1", "rows": [None]}},
    ],
)
def test_unknown_saved_plan_does_not_invent_resource_choices(bad):
    view = ResourceView(resource_report_html(bad or {}))
    assert not view.rows
    assert "supported saved resource table is unavailable" in view.text


def test_missing_requested_fields_do_not_reuse_sizing_suggestions():
    value = plan_view()
    value["plan"]["rows"][0]["requested"] = None
    value["plan"]["rows"][0]["validation"] = None
    value["plan"]["limits"] = None
    view = ResourceView(resource_report_html(value))
    assert view.rows[0][1:] == ["Unavailable"] * 9
    assert "automatic comparison is unavailable" in view.text
    assert "concurrent frequencies: Unavailable" in view.text


@pytest.mark.parametrize(
    "runtime,expected",
    [
        (None, "Unavailable"),
        ("NaN", "Unavailable"),
        (False, "Unavailable"),
        (-1, "Unavailable"),
        (0, "0 s"),
        ("0.25", "0.25 s"),
        (12, "12 s"),
        ("0.123456789012345678901234567891", "0.123456789012345678901234567891 s"),
    ],
)
def test_runtime_is_an_independent_recorded_value(runtime, expected):
    value = plan_view()
    value["plan"]["rows"][0]["estimated_runtime_seconds"] = runtime
    view = ResourceView(resource_report_html(value))
    assert view.rows[0][8] == expected
    assert view.rows[0][7] == "1800"


def test_layout_wall_time_and_memory_changes_are_displayed_without_mutating_data():
    value = plan_view()
    row = value["plan"]["rows"][1]
    row["requested"]["layout"].update(
        nodes=1,
        ranks_per_node=2,
        cpus_per_rank=4,
        node_type="other",
        partition="other-cpu",
    )
    row["requested"]["wall_time_seconds"] = 900
    row["automatic"]["memory_mib_per_node"] = None
    row["validation"]["status"] = "requires_validation"
    view = ResourceView(resource_report_html(value))
    for change in [
        "Nodes: 2 → 1",
        "Ranks per node: 1 → 2",
        "CPUs per rank: 8 → 4",
        "Node type: example-cpu → other",
        "Partition: cpu → other-cpu",
        "Wall-time limit: 1800 → 900 s",
    ]:
        assert change in view.text
    assert "Memory per node: None" not in view.text
    assert "Other fields could not be compared" in view.text
    assert view.rows[1][-1] == "requires_validation (recorded)"


def test_labels_and_exact_plan_details_cannot_create_active_html():
    value = plan_view()
    row = value["plan"]["rows"][0]
    row["requested"]["layout"]["node_type"] = '<script>alert("x")</script>'
    row["requested"]["layout"]["partition"] = '<img src="x" onerror="bad">'
    row["reason"] = '<iframe src="x">instructions</iframe>'
    row["validation"]["status"] = "<svg onload='bad'>"
    view = ResourceView(resource_report_html(value))
    assert not {"script", "img", "iframe", "svg"}.intersection(view.tags)
    assert not any(key.startswith("on") for key, _ in view.attributes)
    assert view.rows[0][2] == row["requested"]["layout"]["node_type"]
    assert "instructions" in view.text


def test_new_report_and_executable_notebook_share_the_readable_plan(
    tmp_path, monkeypatch
):
    pytest.importorskip("matplotlib")
    pytest.importorskip("nbformat")
    display = pytest.importorskip("IPython.display")
    context, snapshots, references = data()
    context["plan"].update(plan_view())
    report = study_report(context, snapshots, references)
    before = deepcopy(report)
    document = ResourceView(report_html(report))
    assert document.rows[:2] == ResourceView(resource_report_html(context["plan"])).rows
    notebook = create_study_notebook(tmp_path / "book", context, snapshots, references)
    captured = []
    monkeypatch.setattr(display, "display", captured.append)
    monkeypatch.chdir(notebook.parent)
    namespace = {}
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
    html_outputs = [item.data for item in captured if isinstance(item, display.HTML)]
    assert any(ResourceView(item).rows == document.rows[:2] for item in html_outputs)
    assert report == before
