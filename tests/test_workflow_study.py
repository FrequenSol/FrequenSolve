import copy
import json

import nbformat
import pytest

from frequensolve.workflow.study import (
    create_study_notebook,
    figure_svg,
    model_summary,
    replay_study_notebook,
    report_html,
    study_report,
    verify_study_replay,
)
from tests.test_workflow import snapshot


def data():
    first = snapshot([3 + 4j, 0])
    first["evidence"].update(
        plan_id="fake-plan",
        execution_id="fake-run",
        plan_revision=3,
        frequency_index=0,
        service_attestation=False,
    )
    second = copy.deepcopy(first)
    second["metadata"]["frequency"] = 20
    second["evidence"]["frequency_index"] = 1
    context = {
        "study": {
            "title": "Acoustic <study>",
            "question": "How does pressure change?",
            "success_criterion": "Compare the reference.",
        },
        "plan": {
            "head": {"plan_id": "fake-plan", "execution_id": "fake-run", "revision": 3},
            "scientific_inputs": {
                "dimension": 3,
                "physics": "acoustic",
                "precision": "double",
                "model": {"subdomains": [{"properties": {"vp": {"value": 1.5}}}]},
                "units": {"defaults": {"velocity": "km/s", "length": "km"}},
                "frequencies": [["2", "0"], ["20", "0"]],
            },
        },
        "credits": {
            "observations": [],
            "history_complete": False,
            "captured_credits": None,
        },
    }
    return context, [first, second], [copy.deepcopy(first), copy.deepcopy(second)]


def test_scientific_narrative_estimates_plots_and_honest_coverage():
    context, values, references = data()
    report = study_report(context, values, references)
    assert report["model"]["wavelengths"] == [
        {"frequency_hz": 2, "wavelength_m": 750},
        {"frequency_hz": 20, "wavelength_m": 75},
    ]
    assert report["reports"][0]["magnitude"] == [[5, 0]]
    assert report["reports"][0]["principal_phase_radians"][0][1] is None
    assert report["reports"][1]["comparison"]["relative_l2_difference"] == 0
    assert "<svg" in figure_svg(report["reports"], "geometry")
    document = report_html(report)
    assert "Acoustic &lt;study&gt;" in document
    assert "<script" not in document
    assert "General accuracy" not in document
    assert report["assurance"]["service_attestation"] is False
    changed = copy.deepcopy(context["plan"]["scientific_inputs"])
    changed["units"]["defaults"]["velocity"] = "unspecified"
    assert model_summary(changed)["wavelengths"] == []


def test_study_rejects_wrong_run_or_reference():
    context, values, refs = data()
    context["plan"]["head"]["execution_id"] = "different-run"
    with pytest.raises(ValueError, match="different executions"):
        study_report(context, values, refs)
    context, values, refs = data()
    refs[0]["metadata"]["source_scale"] = 8
    with pytest.raises(ValueError, match="match exactly"):
        study_report(context, values, refs)


def test_complete_notebook_replay_and_changed_evidence(tmp_path):
    context, values, refs = data()
    path = create_study_notebook(tmp_path / "study", context, values, refs)
    notebook = nbformat.read(path, as_version=4)
    assert len(notebook.cells) >= 15
    receipt = replay_study_notebook(path, timeout=90)
    assert receipt["status"] == "complete"
    assert verify_study_replay(path)["valid"]
    report = json.loads((path.parent / "report.json").read_bytes())
    assert report["reports"][0]["magnitude"] == [[5, 0]]
    with pytest.raises(FileExistsError):
        replay_study_notebook(path)
    (path.parent / "study.json").write_text("{}")
    assert not verify_study_replay(path)["valid"]


def test_changed_code_and_failed_kernel_never_claim_success(tmp_path, monkeypatch):
    context, values, refs = data()
    path = create_study_notebook(tmp_path / "edited", context, values, refs)
    notebook = nbformat.read(path, as_version=4)
    notebook.cells[1].source += '\nprint("changed")'
    nbformat.write(notebook, path)
    with pytest.raises(ValueError, match="changed"):
        replay_study_notebook(path)
    path = create_study_notebook(tmp_path / "failure", context, values, refs)

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted kernel")

    monkeypatch.setattr("nbclient.NotebookClient.execute", fail)
    with pytest.raises(RuntimeError):
        replay_study_notebook(path)
    assert (
        json.loads((path.parent / "replay.json").read_bytes())["status"] == "incomplete"
    )
    assert not (path.parent / "report.html").exists()
    assert not verify_study_replay(path)["valid"]
