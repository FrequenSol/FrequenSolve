"""Saved financial observations remain understandable without inventing totals."""

from copy import deepcopy
from html.parser import HTMLParser

import pytest

from frequensolve.workflow.study import credit_report_html, report_html, study_report
from tests.test_workflow_study import data


class FinancialView(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.tags, self.words, self.rows = [], [], []
        self.row, self.cell = [], None
        self.feed(document)

    def handle_starttag(self, tag, _attrs):
        self.tags.append(tag)
        if tag == "tr":
            self.row = []
        elif tag == "td":
            self.cell = []

    def handle_data(self, value):
        self.words.append(value)
        if self.cell is not None:
            self.cell.append(value)

    def handle_endtag(self, tag):
        if tag == "td":
            self.row.append("".join(self.cell))
            self.cell = None
        elif tag == "tr" and self.row:
            self.rows.append(self.row)

    @property
    def text(self):
        return " ".join(self.words)


def financial_record():
    return {
        "history_complete": True,
        "captured_credits": "1.234567890123",
        "settlement_pending": False,
        "observations": [
            {
                "purpose": "planning",
                "quoted_maximum_credits": "12.5000",
                "authorized_maximum_credits": "10.2500",
                "captured_credits": "0",
                "reconciliation": "settled",
                "observed_at": "2026-01-01T00:00:00Z",
            },
            {
                "purpose": "execution",
                "quoted_maximum_credits": "20",
                "authorized_maximum_credits": "19",
                "captured_credits": "1.234567890123",
                "reconciliation": "settled",
                "observed_at": "2026-01-02T00:00:00Z",
            },
        ],
    }


def test_exact_amounts_and_zero_remain_distinct_from_maxima_and_missing_values():
    credits = financial_record()
    before = deepcopy(credits)
    view = FinancialView(credit_report_html(credits))
    assert view.rows == [
        ["planning", "12.5000", "10.2500", "0", "settled", "2026-01-01T00:00:00Z"],
        ["execution", "20", "19", "1.234567890123", "settled", "2026-01-02T00:00:00Z"],
    ]
    assert "1.234567890123 Credits" in view.text
    assert "complete saved history" in view.text
    assert "No pending settlement is reported for the listed stages" in view.text
    assert "saved financial observations, not a live account balance" in view.text
    assert "Chargebee remains authoritative" in view.text
    assert credits == before


@pytest.mark.parametrize("complete", [False, None])
def test_partial_or_unconfirmed_history_never_presents_a_total(complete):
    credits = financial_record()
    credits.update(history_complete=complete, captured_credits="999.9999")
    view = FinancialView(credit_report_html(credits))
    assert "captured total is unavailable" in view.text
    assert "999.9999" not in view.text
    assert "1.234567890123" in view.rows[1]
    assert "Other operations may be missing" in view.text
    assert (
        "history is incomplete" if complete is False else "has not been confirmed"
    ) in view.text


@pytest.mark.parametrize("value", [None, False, "NaN", "Infinity", "not recorded"])
def test_missing_or_invalid_financial_values_are_unavailable_not_zero(value):
    credits = financial_record()
    credits["captured_credits"] = value
    credits["observations"][0].update(
        quoted_maximum_credits=value,
        authorized_maximum_credits=value,
        captured_credits=value,
        reconciliation=None,
        observed_at=None,
    )
    view = FinancialView(credit_report_html(credits))
    assert view.rows[0] == [
        "planning",
        "Unavailable",
        "Unavailable",
        "Unavailable",
        "Unknown",
        "Not recorded",
    ]
    assert "captured total is unavailable" in view.text
    assert "overall settlement state is unavailable" in view.text
    assert "None" not in view.text and "False" not in view.text


@pytest.mark.parametrize("aggregate_pending", [True, False, None])
def test_recorded_pending_stage_cannot_be_presented_as_fully_settled(aggregate_pending):
    credits = financial_record()
    credits["settlement_pending"] = aggregate_pending
    credits["observations"][1].update(captured_credits=None, reconciliation="pending")
    view = FinancialView(credit_report_html(credits))
    assert view.rows[1][3] == "Unavailable" and view.rows[1][4] == "pending"
    assert "pending for one or more recorded stages" in view.text
    assert "Further charges may remain" in view.text
    assert "No pending settlement" not in view.text


@pytest.mark.parametrize("observations", [[], None])
def test_missing_stage_details_do_not_establish_free_or_settled_work(observations):
    credits = {
        "history_complete": True,
        "captured_credits": "0",
        "settlement_pending": False,
        "observations": observations,
    }
    view = FinancialView(credit_report_html(credits))
    assert not view.rows
    assert "No readable stage observations" in view.text
    assert "do not establish that the workflow was free or settled" in view.text


def test_complete_known_zero_capture_remains_known_zero():
    credits = financial_record()
    credits["observations"] = credits["observations"][:1]
    credits["captured_credits"] = "0.0000"
    view = FinancialView(credit_report_html(credits))
    assert "0.0000 Credits" in view.text and view.rows[0][3] == "0"


@pytest.mark.parametrize("observations", [None, [None]])
def test_unreadable_stage_records_cannot_confirm_history_completeness(observations):
    credits = financial_record()
    credits["observations"] = observations
    view = FinancialView(credit_report_html(credits))
    assert "history completeness has not been confirmed" in view.text
    assert "1.234567890123 Credits" not in view.text


def test_customer_labels_and_timestamps_are_text_without_active_html():
    credits = financial_record()
    credits["observations"][0].update(
        purpose='<script>alert("not instructions")</script>',
        reconciliation='<img src="invalid" onerror="alert(1)">',
        observed_at="<iframe>not a timestamp</iframe>",
    )
    view = FinancialView(credit_report_html(credits))
    assert not {"script", "img", "iframe"}.intersection(view.tags)
    assert view.rows[0][0] == credits["observations"][0]["purpose"]
    assert "overall settlement state is unavailable" in view.text


def test_static_report_includes_clear_incomplete_history_without_changing_evidence():
    pytest.importorskip("matplotlib")
    context, snapshots, references = data()
    report = study_report(context, snapshots, references)
    before = deepcopy(report)
    document = report_html(report)
    view = FinancialView(document)
    assert "saved history is incomplete" in view.text
    assert "captured total is unavailable" in view.text
    assert "Captured total: None" not in view.text
    assert "History complete: False" not in view.text
    assert report == before
