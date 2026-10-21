"""A portable scientific narrative assembled from exact local result snapshots.

Narrative, estimates and local execution evidence never replace solver evidence.
This module has no Cloud client, upload or paid operation.
"""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import io
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .analysis import analyze_pressure, read_snapshot
from .storage import canonical

TEMPLATE = "scientific-study-notebook.v1"
DEFAULT_QUESTION = "How does this acoustic model respond at the selected frequencies?"
DEFAULT_CRITERION = (
    "Inspect finite receiver pressures and compare a compatible retained reference "
    "when available. Physical accuracy requires separate assessment."
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_context(path: str | Path, expected: str) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 256_000:
        raise ValueError("Study context must be a bounded regular file")
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError("Study context changed")
    value = json.loads(data)
    if value.get("version") != TEMPLATE:
        raise ValueError("Unsupported study context")
    return value


def model_summary(science: dict) -> dict:
    """Only derive wavelength when the homogeneous model and units are explicit."""
    result = {
        "physics": science.get("physics"),
        "dimension": science.get("dimension"),
        "precision": science.get("precision"),
        "wavelengths": [],
    }
    try:
        layers = science["model"]["subdomains"]
        units = science["units"]["defaults"]
        if len(layers) != 1 or units["velocity"] not in ("m/s", "km/s"):
            return result
        velocity = float(layers[0]["properties"]["vp"]["value"])
        if not math.isfinite(velocity) or velocity <= 0:
            return result
        velocity_m_s = velocity * (1000 if units["velocity"] == "km/s" else 1)
        for frequency, laplace in science["frequencies"]:
            frequency = float(frequency)
            if float(laplace) != 0 or not math.isfinite(frequency) or frequency <= 0:
                continue
            result["wavelengths"].append(
                {"frequency_hz": frequency, "wavelength_m": velocity_m_s / frequency}
            )
        result.update(
            velocity=velocity,
            velocity_units=units["velocity"],
            length_units=units["length"],
        )
    except (KeyError, TypeError, ValueError, IndexError):
        result["wavelengths"] = []
    return result


def study_report(
    context: dict, snapshots: list[dict], references: list[dict | None]
) -> dict:
    if not 1 <= len(snapshots) <= 2 or len(snapshots) != len(references):
        raise ValueError("Choose one or two explicit study frequencies")
    reports = [
        analyze_pressure(value, reference)
        for value, reference in zip(snapshots, references)
    ]
    indices = [r["assurance"]["input"].get("frequency_index") for r in reports]
    if len(indices) != len(set(indices)):
        raise ValueError("Study frequencies must be unique")
    head = context["plan"]["head"]
    if len({r["assurance"]["input"].get("result_digest") for r in reports}) != 1:
        raise ValueError("Study results refer to different execution manifests")
    for report in reports:
        evidence = report["assurance"]["input"]
        if (
            evidence.get("plan_id") != head["plan_id"]
            or evidence.get("execution_id") != head.get("execution_id")
            or evidence.get("plan_revision") != head["revision"]
        ):
            raise ValueError("Study context and results refer to different executions")
    return {
        "version": "scientific-study-report.v1",
        "study": context["study"],
        "model": model_summary(context["plan"].get("scientific_inputs", {})),
        "plan": context["plan"],
        "credits": context["credits"],
        "reports": reports,
        "assurance": {
            "origin": "local",
            "service_attestation": False,
            "scientific_claim": "Reference agreement and solver convergence do not establish general physical or discretization accuracy.",
        },
    }


def figure_svg(reports: list[dict], kind: str = "pressure") -> str:
    """Deterministic discrete scientific figures; never interpolate between frequencies."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context(
        {"svg.fonttype": "none", "svg.hashsalt": TEMPLATE, "font.size": 10}
    ):
        if kind == "geometry":
            fig = plt.figure(figsize=(8, 4.5), layout="constrained")
            ax = fig.add_subplot(111, projection="3d")
            meta = reports[0]["metadata"]
            for name, marker, color in (
                ("source", "*", "#b45309"),
                ("receiver", "o", "#16756d"),
            ):
                xyz = np.asarray(meta[name + "_coordinates"], dtype=float)
                if xyz.shape[1] != 3:
                    raise ValueError("This study requires explicit 3D geometry")
                ax.scatter(
                    *xyz.T,
                    marker=marker,
                    color=color,
                    s=90 if name == "source" else 32,
                    label=name.title(),
                )
                for label, point in zip(meta[name + "_ids"], xyz):
                    ax.text(*point, f" {name[0].upper()}{label}", fontsize=8)
            for axis in "xyz":
                getattr(ax, f"set_{axis}label")(
                    f"{axis.upper()} ({meta['coordinate_units']})"
                )
            ax.set_title("Source and receiver geometry · native coordinates")
            ax.legend(loc="upper left")
        else:
            fields = [
                ("real", "Real pressure"),
                ("imaginary", "Imaginary pressure"),
                ("magnitude", "Magnitude"),
                ("principal_phase_radians", "Principal phase"),
            ]
            fig, axes = plt.subplots(
                len(reports),
                4,
                figsize=(13, max(3, 3 * len(reports))),
                squeeze=False,
                layout="constrained",
            )
            for row, report in enumerate(reports):
                meta = report["metadata"]
                labels = [
                    f"S{s} → R{r}"
                    for s in meta["source_ids"]
                    for r in meta["receiver_ids"]
                ]
                for col, (field, title) in enumerate(fields):
                    ax = axes[row, col]
                    values = np.array(report[field], dtype=float).reshape(-1)
                    ax.scatter(range(len(values)), values, color="#16756d", s=36)
                    ax.set_xticks(range(len(values)), labels, rotation=35, ha="right")
                    ax.set_title(f"{meta['frequency']} Hz · {title}")
                    ax.set_ylabel(
                        "radians"
                        if field == "principal_phase_radians"
                        else meta["units"]
                    )
                    ax.grid(axis="y", alpha=0.2)
                    if not np.isfinite(values).all():
                        ax.text(
                            0.02,
                            0.96,
                            "Missing/undefined values omitted",
                            transform=ax.transAxes,
                            va="top",
                            fontsize=8,
                        )
        output = io.StringIO()
        fig.savefig(output, format="svg", metadata={"Date": None})
        plt.close(fig)
    # Generated by Matplotlib from bounded numeric arrays and escaped labels.
    return output.getvalue()[output.getvalue().index("<svg") :]


def report_html(report: dict) -> str:
    esc = lambda value: html.escape(str(value))
    study, model = report["study"], report["model"]
    science = report["plan"].get("scientific_inputs", {})
    rows = "".join(
        f"<tr><td>{esc(r['metadata']['frequency'])} Hz</td><td>{r['finite_count']} / {r['sample_count']}</td><td>{esc(r.get('comparison', {}).get('coverage', 'Not compared'))}</td><td>{esc(r.get('comparison', {}).get('relative_l2_difference', 'Not available'))}</td></tr>"
        for r in report["reports"]
    )
    wavelengths = "".join(
        f"<li>{x['frequency_hz']:g} Hz: {x['wavelength_m']:g} m</li>"
        for x in model["wavelengths"]
    )
    credits = report["credits"]
    charges = "".join(
        f"<tr><td>{esc(x.get('purpose'))}</td><td>{esc(x.get('quoted_maximum_credits', 'Unavailable'))}</td><td>{esc(x.get('captured_credits', 'Unsettled'))}</td><td>{esc(x.get('reconciliation', 'Unknown'))}</td></tr>"
        for x in credits.get("observations", [])
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(study['title'])}</title>
<style>body{{font:16px/1.65 system-ui;color:#20333c;max-width:1100px;margin:auto;padding:40px 28px;background:#fcfdfd}}h1{{font-size:36px;line-height:1.2}}h2{{margin-top:40px}}svg{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #dce5e8;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f5;padding:16px}}.note{{border-left:4px solid #16756d;padding:12px 20px;background:#edf7f4}}</style>
<h1>{esc(study['title'])}</h1><p>Scientific study · analysis reproduced locally</p>
<h2>1. Question and success criteria</h2><p>{esc(study['question'])}</p><p><strong>Success criterion:</strong> {esc(study['success_criterion'])}</p>
<h2>2. Model and assumptions</h2><p>{esc(science.get('dimension', 'Unknown'))}D {esc(science.get('physics', 'unknown physics'))}, {esc(science.get('precision', 'unknown'))} precision. Scientific inputs and source scaling are retained below.</p>
{figure_svg(report['reports'], 'geometry')}<details><summary>Inspect materials, boundaries, mesh, units and solver settings</summary><pre>{esc(json.dumps(science, indent=2))}</pre></details>
<h2>3. Estimates before interpreting the result</h2><p>For an explicitly homogeneous velocity, wavelength = velocity / frequency.</p><ul>{wavelengths or '<li>No supported wavelength estimate is available from these inputs.</li>'}</ul><p>This scale estimate does not establish mesh or boundary accuracy. Frequencies remain independent.</p>
<h2>4. Decisions, execution and Credits</h2><p>Planning, paid validation when necessary, and execution have separate approvals. This document records existing work and cannot submit another run.</p><table><tr><th>Stage</th><th>Quoted maximum (Credits)</th><th>Captured (Credits)</th><th>Settlement</th></tr>{charges}</table><p>History complete: {esc(credits.get('history_complete', False))}. Captured total: {esc(credits.get('captured_credits', 'Unavailable'))} Credits. Internal reservations are not Chargebee holds.</p><details><summary>Inspect exact resource choices and recorded revisions</summary><pre>{esc(json.dumps(report['plan'].get('plan', {}), indent=2))}</pre></details>
<h2>5. Receiver response and comparison</h2>{figure_svg(report['reports'])}<table><tr><th>Frequency</th><th>Finite / selected samples</th><th>Paired-reference coverage</th><th>Relative L2 difference</th></tr>{rows}</table><p>Phase is principal phase in radians and is undefined at zero magnitude. No interpolation, rescaling, unwrapping or time reconstruction is performed.</p>
<h2>6. Conclusions and limits</h2><p>{sum(r['finite_count'] for r in report['reports'])} of {sum(r['sample_count'] for r in report['reports'])} selected values are finite. Comparisons above use only exactly compatible metadata. An absent reference is not a passed comparison.</p><div class="note">{esc(report['assurance']['scientific_claim'])} No engineering acceptance threshold is inferred. Measured distributed peak memory is not supplied by this analysis.</div>
<h2>7. Reproduce and continue</h2><p>Open analysis.ipynb with the bundled inputs and the recorded Python environment. Run All performs local analysis only. Edit a copy to explore another interpretation; edits invalidate the original execution evidence. Preparing another simulation requires its own reviewed workflow.</p><details><summary>Input, execution and analysis provenance</summary><pre>{esc(json.dumps({'head': report['plan']['head'], 'assurance': report['assurance'], 'inputs': [r['assurance'] for r in report['reports']]}, indent=2))}</pre></details></html>"""


def notebook_cells(context_hash: str, entries: list[dict]) -> list[tuple[str, str]]:
    reads = [
        "from frequensolve.workflow.study import read_context, study_report, figure_svg, report_html",
        "from frequensolve.workflow.analysis import read_snapshot",
        "from IPython.display import SVG, HTML, display",
        "import json",
        f"context = read_context('study.json', {context_hash!r})",
        "snapshots, references = [], []",
    ]
    for item in entries:
        target = "snapshots" if item["kind"] == "snapshot" else "references"
        reads.append(
            f"{target}.append(read_snapshot({item['path']!r}, {item['sha256']!r}))"
        )
        if item.get("no_reference"):
            reads.append("references.append(None)")
    return [
        (
            "markdown",
            "# A reproducible scientific study\n\nRun All analyzes verified local data. It never uploads, plans, authorizes or submits compute. Edit a copy for custom work; previous execution evidence does not apply after edits.",
        ),
        ("code", "\n".join(reads)),
        (
            "markdown",
            "## 1. Question and success criteria\nThe question is customer-authored context, not a scientific validation result.",
        ),
        (
            "code",
            "print(context['study']['title'])\nprint(context['study']['question'])\nprint('Success criterion:', context['study']['success_criterion'])",
        ),
        (
            "markdown",
            "## 2. Model, assumptions and geometry\nInspect units, materials, boundaries, sources, receivers and numerical settings before interpreting pressure.",
        ),
        (
            "code",
            "print(json.dumps(context['plan'].get('scientific_inputs', {}), indent=2))\nreport = study_report(context, snapshots, references)\ndisplay(SVG(figure_svg(report['reports'], 'geometry')))",
        ),
        (
            "markdown",
            "## 3. First-order estimate\nWavelength = velocity / frequency for a homogeneous model. This is a scale estimate, not a mesh-convergence test.",
        ),
        ("code", "print(json.dumps(report['model'], indent=2))"),
        (
            "markdown",
            "## 4. Plan choices, progress and Credits\nThese are retained observations of existing operations. Quotes, maxima, captured charges and settlement are distinct.",
        ),
        (
            "code",
            "print(json.dumps(context['plan'].get('plan', {}), indent=2))\nprint(json.dumps(context['credits'], indent=2))",
        ),
        (
            "markdown",
            "## 5. Complex receiver pressure\nDiscrete native values only. Zero-magnitude phase is undefined. No interpolation or phase unwrapping is performed.",
        ),
        ("code", "display(SVG(figure_svg(report['reports'])))"),
        (
            "markdown",
            "## 6. Comparison, conclusions and limitations\nReference alignment requires matching source/receiver identity, coordinates, units, frequency, Laplace coordinate and source scaling. Convergence and reference agreement do not establish general physical accuracy.",
        ),
        (
            "code",
            "for result in report['reports']:\n    print('Frequency:', result['metadata']['frequency'], 'Hz; finite:', result['finite_count'], '/', result['sample_count'])\n    print('Reference:', result.get('comparison', 'No compatible reference selected'))\nprint(report['assurance']['scientific_claim'])",
        ),
        (
            "markdown",
            "## 7. Portable report and provenance\nThe report below is generated from the same verified values. Local replay evidence is not service attestation. Use a separate approved workflow to run another simulation.",
        ),
        ("code", "display(HTML(report_html(report)))"),
        ("code", "print(json.dumps(report, sort_keys=True, allow_nan=False))"),
    ]


def create_study_notebook(
    destination: Path,
    context: dict,
    snapshots: list[dict],
    references: list[dict | None],
) -> Path:
    import nbformat

    destination = Path(destination)
    study_report(context, snapshots, references)
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    context = {**context, "version": TEMPLATE}
    context_bytes = canonical(context)
    if len(context_bytes) > 256_000:
        raise ValueError("Study context exceeds the portable limit")
    (destination / "study.json").write_bytes(context_bytes)
    entries = []
    for index, (snapshot, reference) in enumerate(zip(snapshots, references)):
        for kind, data in (("snapshot", snapshot), ("reference", reference)):
            if data is None:
                continue
            name, content = f"{kind}-{index}.json", canonical(data)
            (destination / name).write_bytes(content)
            entries.append(
                {
                    "kind": kind,
                    "path": name,
                    "sha256": sha(content),
                    "no_reference": kind == "snapshot" and reference is None,
                }
            )
    cells = notebook_cells(sha(context_bytes), entries)
    notebook = nbformat.v4.new_notebook(
        cells=[
            (
                nbformat.v4.new_code_cell(source)
                if kind == "code"
                else nbformat.v4.new_markdown_cell(source)
            )
            for kind, source in cells
        ],
        metadata={
            "frequensolve_study": {
                "template": TEMPLATE,
                "context_sha256": sha(context_bytes),
                "inputs": entries,
                "code_sha256": sha(canonical([s for k, s in cells if k == "code"])),
            }
        },
    )
    path = destination / "analysis.ipynb"
    path.write_text(nbformat.writes(notebook))
    return path


def replay_study_notebook(path: Path, timeout: int = 120) -> dict:
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 1_000_000 or not 1 <= timeout <= 600:
        raise ValueError("Invalid study notebook")
    notebook = nbformat.read(path, as_version=4)
    meta = notebook.metadata["frequensolve_study"]
    if meta["template"] != TEMPLATE or not 1 <= len(meta["inputs"]) <= 4:
        raise ValueError("Unsupported study template")
    for item in meta["inputs"]:
        if item["kind"] not in ("snapshot", "reference") or item["path"] not in [
            f"{item['kind']}-{n}.json" for n in (0, 1)
        ]:
            raise ValueError("Unsupported study input path")
    if len({item["path"] for item in meta["inputs"]}) != len(meta["inputs"]):
        raise ValueError("Duplicate study inputs")
    expected = notebook_cells(meta["context_sha256"], meta["inputs"])
    code = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    if (
        code != [s for k, s in expected if k == "code"]
        or sha(canonical(code)) != meta["code_sha256"]
    ):
        raise ValueError(
            "Study code changed; use Jupyter for custom analysis. Original replay evidence no longer applies."
        )
    context = read_context(path.parent / "study.json", meta["context_sha256"])
    snapshots, references = [], []
    for item in meta["inputs"]:
        value = read_snapshot(path.parent / item["path"], item["sha256"])
        (snapshots if item["kind"] == "snapshot" else references).append(value)
        if item["no_reference"]:
            references.append(None)
    report = study_report(context, snapshots, references)
    files = (
        "executed.ipynb",
        "report.json",
        "report.html",
        "geometry.svg",
        "pressure.svg",
        "replay.json",
    )
    if any(
        (path.parent / name).exists() or (path.parent / name).is_symlink()
        for name in files
    ):
        raise FileExistsError(
            "Preserve previous study execution; choose a new directory"
        )
    receipt = {
        "version": "study-replay.v1",
        "status": "incomplete",
        "origin": "local",
        "service_attestation": False,
        "at": datetime.now(timezone.utc).isoformat(),
        "notebook_sha256": sha(path.read_bytes()),
        "context_sha256": meta["context_sha256"],
        "inputs": meta["inputs"],
        "python": sys.version.split()[0],
        "interpreter": sys.executable,
        "dependencies": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "matplotlib", "nbformat", "nbclient", "ipykernel")
        },
        "operation_source_sha256": {
            n: sha(Path(__file__).with_name(n).read_bytes())
            for n in ("study.py", "analysis.py")
        },
    }
    (path.parent / "replay.json").write_bytes(canonical(receipt))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    try:
        with tempfile.TemporaryDirectory(prefix="fs-study-kernel-") as temp:
            kernels = Path(temp) / "kernels"
            spec = kernels / "frequensolve-study"
            spec.mkdir(parents=True)
            (spec / "kernel.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            sys.executable,
                            "-m",
                            "ipykernel_launcher",
                            "-f",
                            "{connection_file}",
                        ],
                        "display_name": "FrequenSolve study",
                        "language": "python",
                    }
                )
            )
            manager = KernelManager(
                kernel_name="frequensolve-study",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
            )
            env = {
                k: os.environ[k]
                for k in ("PATH", "SYSTEMROOT", "TMPDIR")
                if k in os.environ
            }
            env.update(
                HOME=temp,
                IPYTHONDIR=temp,
                MPLCONFIGDIR=temp,
                JUPYTER_CONFIG_DIR=temp,
                PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                PYTHONNOUSERSITE="1",
            )
            NotebookClient(
                notebook, km=manager, timeout=timeout, allow_errors=False
            ).execute(cwd=str(path.parent.absolute()), env=env, cleanup_kc=True)
        stdout = "".join(
            o.get("text", "")
            for o in notebook.cells[-1].get("outputs", [])
            if o.get("output_type") == "stream" and o.get("name") == "stdout"
        )
        if canonical(json.loads(stdout)) != canonical(report):
            raise ValueError(
                "Notebook calculations differ from the tested study operation"
            )
        (path.parent / "report.json").write_bytes(canonical(report))
        (path.parent / "report.html").write_text(report_html(report))
        for kind in ("geometry", "pressure"):
            (path.parent / f"{kind}.svg").write_text(
                figure_svg(report["reports"], kind)
            )
        receipt["status"] = "complete"
    finally:
        if receipt["status"] != "complete":
            for name in ("report.json", "report.html", "geometry.svg", "pressure.svg"):
                (path.parent / name).unlink(missing_ok=True)
        (path.parent / "executed.ipynb").write_text(nbformat.writes(notebook))
        receipt["outputs"] = {
            name: sha((path.parent / name).read_bytes())
            for name in files
            if name != "replay.json" and (path.parent / name).is_file()
        }
        (path.parent / "replay.json").write_bytes(canonical(receipt))
    return receipt


def verify_study_replay(path: Path) -> dict:
    path = Path(path)
    changes = []
    try:
        evidence_file = path.parent / "replay.json"
        if (
            evidence_file.is_symlink()
            or not evidence_file.is_file()
            or evidence_file.stat().st_size > 64000
        ):
            raise ValueError("Invalid replay evidence")
        receipt = json.loads(evidence_file.read_bytes())
        if (
            receipt.get("version") != "study-replay.v1"
            or receipt.get("status") != "complete"
            or receipt.get("origin") != "local"
            or receipt.get("service_attestation") is not False
        ):
            raise ValueError("Replay incomplete")
        if (
            set(receipt["outputs"])
            != {
                "executed.ipynb",
                "report.json",
                "report.html",
                "geometry.svg",
                "pressure.svg",
            }
            or set(receipt["dependencies"])
            != {"numpy", "matplotlib", "nbformat", "nbclient", "ipykernel"}
            or set(receipt["operation_source_sha256"]) != {"study.py", "analysis.py"}
        ):
            raise ValueError("Replay evidence is incomplete")
        if not 1 <= len(receipt["inputs"]) <= 4 or len(
            {x["path"] for x in receipt["inputs"]}
        ) != len(receipt["inputs"]):
            raise ValueError("Invalid replay inputs")
        for item in receipt["inputs"]:
            if item["kind"] not in ("snapshot", "reference") or item["path"] not in [
                f"{item['kind']}-{n}.json" for n in (0, 1)
            ]:
                raise ValueError("Invalid replay input identity")
        expected = {
            path.name: receipt["notebook_sha256"],
            "study.json": receipt["context_sha256"],
            **{x["path"]: x["sha256"] for x in receipt["inputs"]},
            **receipt["outputs"],
        }
        for name, digest in expected.items():
            if (
                Path(name).name != name
                or (path.parent / name).is_symlink()
                or not (path.parent / name).is_file()
                or (path.parent / name).stat().st_size > 16 * 1024 * 1024
                or sha((path.parent / name).read_bytes()) != digest
            ):
                changes.append("input_code_or_output_changed")
        if (
            receipt["python"] != sys.version.split()[0]
            or receipt["interpreter"] != sys.executable
            or any(
                importlib.metadata.version(n) != v
                for n, v in receipt["dependencies"].items()
            )
        ):
            changes.append("environment_changed")
        if any(
            n not in ("study.py", "analysis.py")
            or sha(Path(__file__).with_name(n).read_bytes()) != v
            for n, v in receipt["operation_source_sha256"].items()
        ):
            changes.append("operation_changed")
    except Exception:
        changes.append("evidence_unavailable_or_incomplete")
    return {
        "valid": not changes,
        "origin": "local",
        "service_attestation": False,
        "mismatches": sorted(set(changes)),
    }
