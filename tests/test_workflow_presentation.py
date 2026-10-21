"""Scientific conversation content, read authority and standard MCP transport."""

import base64
import copy
import json

import anyio
import pytest
from mcp import Client
from mcp.types import ImageContent, TextContent

from frequensolve.mcp_server._sdk_v2 import _bounded_content
from frequensolve.workflow.analysis import analyze_pressure
from frequensolve.workflow.client import create_project
from frequensolve.workflow.presentation import geometry, present
from frequensolve.workflow.server import build_server
from tests.test_workflow_study import data


def view():
    context, snapshots, references = data()
    context["plan"]["head"]["status"] = "executed"
    context["plan"]["input_digest"] = "sha256:" + "1" * 64
    context["plan"]["plan"] = {"digest": "sha256:" + "2" * 64}
    context["plan"]["execution"] = {"outcome": "FAILED", "phase": "settling"}
    science = context["plan"]["scientific_inputs"]
    science["mesh"] = {"generator": {"l_bound": [0, 0, 0], "u_bound": [1, 2, 1]}}
    science["acquisition"] = {
        "source_geometry": {"sources": [{"coordinates": [0.25, 0.5, 0.2]}]},
        "receiver_groups": [
            {"coordinates": {"units": "km", "value": [[0.75, 0.5, 0.2]]}}
        ],
    }
    for item in snapshots:
        item["evidence"]["result_digest"] = "sha256:" + "3" * 64
    context["study"].update(
        analysis_origin="local_client_reported",
        service_attestation=False,
        analysis={
            "execution_id": "fake-run",
            "plan_revision": 3,
            "input_digest": context["plan"]["input_digest"],
            "plan_digest": context["plan"]["plan"]["digest"],
            "result_digest": "sha256:" + "3" * 64,
            "reports": [analyze_pressure(s, r) for s, r in zip(snapshots, references)],
        },
    )
    return context


def test_recalculates_arrays_with_zero_phase_and_explicit_recorded_comparison():
    value = view()
    # A stale derived magnitude must not change the picture or reported science.
    value["study"]["analysis"]["reports"][0]["magnitude"] = [[999, 999]]
    output = present(value, "results", [0, 1], figures=False)
    report = output.provenance["reports"][0]
    assert report["magnitude"] == [[5, 0]]
    assert report["principal_phase_radians"][0][1] is None
    assert report["recorded_comparison"] is None
    assert "reference comparison is withheld" in output.narrative
    assert "not been independently reverified" in output.narrative
    assert not output.provenance["service_attestation"]
    assert not output.provenance["input_files_reverified"]
    assert not output.provenance["paid_request_sent"]
    assert not output.figures
    checked = present(view(), "results", [0, 1], figures=False)
    assert (
        checked.provenance["reports"][0]["recorded_comparison"][
            "relative_l2_difference"
        ]
        == 0
    )
    assert "Recorded reference comparison" in checked.narrative


@pytest.mark.parametrize(
    "key,value",
    [
        ("input_digest", "different"),
        ("plan_digest", "different"),
        ("plan_revision", 4),
        ("execution_id", "other-run"),
    ],
)
def test_stale_analysis_never_draws_current_results(key, value):
    current = view()
    current["study"]["analysis"][key] = value
    output = present(current, "results", [0, 1])
    assert output.provenance["reports"] == []
    assert output.provenance["missing_frequency_indices"] == [0, 1]
    assert not output.figures
    assert "earlier or different" in output.narrative


def test_missing_nonfinite_and_nonconvergence_remain_explicit():
    value = view()
    value["study"]["analysis"]["reports"] = value["study"]["analysis"]["reports"][:1]
    value["study"]["analysis"]["reports"][0]["real"][0][0] = None
    output = present(value, "results", [0, 1], False)
    assert output.provenance["missing_frequency_indices"] == [1]
    assert "1 / 2 finite" in output.narrative
    value["study"]["analysis"]["reports"][0]["assurance"]["input"]["converged"] = False
    with pytest.raises(ValueError, match="converged"):
        present(value, "results", [0, 1], False)


def test_progress_does_not_conflate_terminal_plan_science_publication_or_settlement():
    output = present(view(), "progress", [0], False)
    assert "executed" in output.narrative and "FAILED" in output.narrative
    assert (
        "settling" in output.narrative and "history is incomplete" in output.narrative
    )
    assert "download availability is not checked" in output.narrative
    assert "Chargebee remains authoritative" in output.narrative
    broken = view()
    broken["study"]["analysis"]["reports"][0]["assurance"]["input"]["converged"] = False
    unverified = present(broken, "progress", [0], False)
    assert "could not be verified" in unverified.narrative
    assert "FAILED" in unverified.narrative and "settling" in unverified.narrative


def test_frequency_identity_cannot_change_under_an_existing_run_binding():
    value = view()
    value["study"]["analysis"]["reports"][0]["metadata"]["frequency"] = 17
    with pytest.raises(ValueError, match="frequency coordinates"):
        present(value, "results", [0], False)


def test_pre_result_model_figure_uses_only_explicit_geometry_and_units():
    pytest.importorskip("matplotlib")
    value = view()
    value["study"].pop("analysis")
    output = present(value, "model", [0, 1])
    assert output.figures[0].startswith(b"\x89PNG\r\n\x1a\n")
    assert len(output.figures[0]) < 1_048_576
    assert "750 m" in output.narrative and "75 m" in output.narrative
    assert "not a simulated wavefield" in output.narrative
    changed = copy.deepcopy(value["plan"]["scientific_inputs"])
    changed["acquisition"]["receiver_groups"][0]["coordinates"]["units"] = "m"
    assert geometry(changed) is None  # Never silently convert or mislabel coordinates.
    changed["mesh"]["generator"]["u_bound"][0] = None
    assert geometry(changed) is None


def test_stored_text_cannot_embed_html_or_markdown_links():
    value = view()
    value["study"]["title"] = "<script>alert(1)</script> [click](https://evil.test)"
    result = present(value, "overview", [0], False)
    assert "<script>" not in result.narrative
    assert "[click](https://evil.test)" not in result.narrative


def test_protocol_returns_human_text_and_png_with_separate_bounded_provenance(
    tmp_path, monkeypatch
):
    pytest.importorskip("matplotlib")
    state = view()
    calls = []
    create_project(tmp_path, "study", "A scientific question")

    class Reader:
        def call(self, operation, arguments):
            calls.append(operation)
            if operation == "plan_get":
                return state["plan"]
            assert operation == "study_get", "Presentation attempted a mutation"
            return {"study": state["study"]}

    monkeypatch.setattr(
        "frequensolve.workflow.server.credit_scan", lambda *_: state["credits"]
    )

    async def run():
        async with Client(
            build_server(Reader(), {"root": tmp_path}, "https://example.test"),
            cache=None,
        ) as client:
            tools = (await client.list_tools()).tools
            tool = next(t for t in tools if t.name == "workflow_present")
            assert tool.annotations.read_only_hint
            assert "workflow_approve" not in {t.name for t in tools}
            result = await client.call_tool(
                "workflow_present",
                {
                    "request": {
                        "root_id": "root",
                        "local_project": "study",
                        "plan_id": "fake-plan",
                        "chapter": "results",
                    }
                },
            )
            assert not result.is_error and result.structured_content["ok"]
            assert len(result.content) == 2
            assert isinstance(result.content[0], TextContent)
            assert result.content[0].text.startswith("**Acoustic")
            assert isinstance(result.content[1], ImageContent)
            assert result.content[1].mime_type == "image/png"
            assert base64.b64decode(result.content[1].data).startswith(b"\x89PNG")
            assert result.content[1].data not in json.dumps(result.structured_content)
            assert len(json.dumps(result.structured_content)) < 32_768
            denied = await client.call_tool(
                "workflow_present",
                {
                    "request": {
                        "root_id": "root",
                        "local_project": "study",
                        "plan_id": "fake-plan",
                        "stage": "execution",
                        "approve": True,
                    }
                },
            )
            assert denied.is_error

    anyio.run(run)
    assert calls == ["plan_get", "study_get"]


@pytest.mark.parametrize(
    "content",
    [
        [],
        [TextContent(type="text", text="x" * 32_769)],
        [TextContent(type="text", text="x")] * 4,
        [ImageContent(type="image", mime_type="image/svg+xml", data="PHN2Zz4=")],
        [
            ImageContent(
                type="image",
                mime_type="image/png",
                data=base64.b64encode(b"not an image").decode(),
            )
        ],
        [ImageContent(type="image", mime_type="image/png", data="x" * 1_400_001)],
    ],
)
def test_content_budget_rejects_unbounded_or_active_content(content):
    with pytest.raises(ValueError):
        _bounded_content(content)
