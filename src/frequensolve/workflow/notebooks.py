"""Portable analysis-only notebook bundles and clean local replay evidence."""

from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .analysis import analyze_pressure, read_snapshot
from .storage import canonical

TEMPLATE = "pressure-analysis-notebook.v1"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code(inputs: list[dict]) -> str:
    lines = [
        "from frequensolve.workflow.analysis import read_snapshot, analyze_pressure",
        "import json",
    ]
    for index, item in enumerate(inputs):
        lines.append(
            f"snapshot_{index} = read_snapshot({item['path']!r}, {item['sha256']!r})"
        )
    lines += [
        "report = analyze_pressure(snapshot_0"
        + (", snapshot_1" if len(inputs) == 2 else "")
        + ")",
        "print(json.dumps(report, indent=2, allow_nan=False))",
    ]
    return "\n".join(lines)


def create_notebook(
    directory: Path, snapshot: Path, reference: Path | None = None
) -> Path:
    import nbformat

    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=False, exist_ok=False)
    inputs = []
    for index, path in enumerate(
        [snapshot] + ([reference] if reference is not None else [])
    ):
        data = Path(path).read_bytes()
        digest = _digest(data)
        read_snapshot(path, digest)
        name = f"pressure-{index}.json"
        (directory / name).write_bytes(data)
        inputs.append({"path": name, "sha256": digest})
    code = _code(inputs)
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# Receiver pressure analysis\n\nThis notebook analyzes verified local snapshots. Run all performs no Cloud operation.\n\n"
                "Principal phase is undefined at zero magnitude. Frequencies remain independent. "
                "A locally rerun calculation does not establish scientific correctness."
            ),
            nbformat.v4.new_code_cell(code),
        ],
        metadata={
            "frequensolve_workflow": {
                "template": TEMPLATE,
                "inputs": inputs,
                "code_sha256": _digest(code.encode()),
                "assurance": "generated; not yet rerun",
            }
        },
    )
    path = directory / "analysis.ipynb"
    path.write_text(nbformat.writes(notebook))
    return path


def replay_notebook(path: Path, *, timeout: int = 120) -> dict:
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    path = Path(path)
    if (
        path.is_symlink()
        or path.stat().st_size > 1024 * 1024
        or not 1 <= timeout <= 600
    ):
        raise ValueError("Invalid generated notebook")
    notebook = nbformat.read(path, as_version=4)
    meta = notebook.metadata.get("frequensolve_workflow", {})
    inputs = meta.get("inputs", [])
    if (
        meta.get("template") != TEMPLATE
        or len(inputs) not in (1, 2)
        or any(
            set(item) != {"path", "sha256"} or item["path"] != f"pressure-{index}.json"
            for index, item in enumerate(inputs)
        )
    ):
        raise ValueError(
            "Only generated analysis templates can receive official replay evidence"
        )
    code = _code(inputs)
    if [cell.source for cell in notebook.cells if cell.cell_type == "code"] != [
        code
    ] or meta.get("code_sha256") != _digest(code.encode()):
        raise ValueError(
            "Notebook code was edited; use Jupyter for custom local work or generate a fresh official analysis"
        )
    snapshots = [
        read_snapshot(path.parent / item["path"], item["sha256"]) for item in inputs
    ]
    result = analyze_pressure(*snapshots)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    receipt = {
        "version": "local-notebook-replay.v1",
        "status": "incomplete",
        "origin": "local",
        "notebook_sha256": _digest(path.read_bytes()),
        "code_sha256": meta["code_sha256"],
        "inputs": inputs,
        "interpreter": sys.executable,
        "python": sys.version.split()[0],
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "nbclient", "nbformat", "ipykernel")
        },
        "operation_source_sha256": _digest(
            Path(__file__).with_name("analysis.py").read_bytes()
        ),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    # Fixed new output locations preserve previous executions and custom edits.
    for name in ("executed.ipynb", "replay.json", "report.json", "report.html"):
        if (path.parent / name).exists() or (path.parent / name).is_symlink():
            raise FileExistsError("Replay outputs already exist; create a new bundle")
    try:
        with tempfile.TemporaryDirectory(prefix="fs-analysis-kernel-") as temp:
            kernels = Path(temp) / "kernels"
            spec = kernels / "frequensolve-workflow"
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
                        "display_name": "FrequenSolve analysis",
                        "language": "python",
                    }
                )
            )
            manager = KernelManager(
                kernel_name="frequensolve-workflow",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
            )
            env = {
                key: os.environ[key]
                for key in ("PATH", "SYSTEMROOT", "TMPDIR")
                if key in os.environ
            }
            env.update(
                HOME=temp,
                IPYTHONDIR=temp,
                JUPYTER_CONFIG_DIR=temp,
                PYTHONPATH=str(Path(__file__).resolve().parents[2]),
                PYTHONNOUSERSITE="1",
            )
            NotebookClient(
                notebook, km=manager, timeout=timeout, allow_errors=False
            ).execute(cwd=str(path.parent.absolute()), env=env, cleanup_kc=True)
        output = "".join(
            item.get("text", "")
            for cell in notebook.cells
            if cell.cell_type == "code"
            for item in cell.get("outputs", [])
            if item.get("output_type") == "stream" and item.get("name") == "stdout"
        )
        if canonical(json.loads(output)) != canonical(result):
            raise ValueError(
                "Kernel output differs from the inspected analysis operation"
            )
        receipt["status"] = "complete"
        receipt["report_sha256"] = _digest(canonical(result))
        (path.parent / "report.json").write_bytes(canonical(result))
        # Only escaped, static text; no imported notebook HTML, scripts or widgets.
        (path.parent / "report.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Pressure analysis</title>'
            "<h1>Pressure analysis</h1><p>Locally rerun. Scientific interpretation requires review.</p><pre>"
            + html.escape(json.dumps(result, indent=2, allow_nan=False))
            + "</pre>"
        )
        receipt["report_html_sha256"] = _digest(
            (path.parent / "report.html").read_bytes()
        )
    finally:
        executed = nbformat.writes(notebook).encode()
        receipt["executed_notebook_sha256"] = _digest(executed)
        (path.parent / "executed.ipynb").write_bytes(executed)
        (path.parent / "replay.json").write_bytes(canonical(receipt))
    return receipt


def verify_replay(path: Path) -> dict:
    """Check whether retained local execution evidence still applies here.

    Historical receipts remain unchanged. Editing a notebook/input/output or
    changing its environment invalidates the current claim, not that history.
    This checks identities; it is not service or scientific attestation.
    """
    import nbformat

    path = Path(path)
    mismatches = []

    def bounded(name: str, maximum: int = 16 * 1024 * 1024) -> bytes:
        file = path.parent / name
        if file.is_symlink() or not file.is_file() or file.stat().st_size > maximum:
            raise ValueError("Missing or unsupported local evidence")
        return file.read_bytes()

    try:
        receipt = json.loads(bounded("replay.json", 2 * 1024 * 1024))
        if receipt.get("status") != "complete" or receipt.get("origin") != "local":
            return {
                "valid": False,
                "origin": "local",
                "mismatches": ["execution_incomplete"],
            }
        original = bounded(path.name, 1024 * 1024)
        if _digest(original) != receipt["notebook_sha256"]:
            mismatches.append("notebook_changed")
        notebook = nbformat.reads(original.decode(), as_version=4)
        inputs = notebook.metadata["frequensolve_workflow"]["inputs"]
        if inputs != receipt["inputs"] or any(
            item.get("path") != f"pressure-{index}.json"
            for index, item in enumerate(inputs)
        ):
            raise ValueError("Input identity changed")
        if len(inputs) not in (1, 2):
            raise ValueError("Unsupported input selection")
        for item in inputs:
            if _digest(bounded(item["path"])) != item["sha256"]:
                mismatches.append("input_changed")
        if (
            _digest(Path(__file__).with_name("analysis.py").read_bytes())
            != receipt["operation_source_sha256"]
        ):
            mismatches.append("operation_changed")
        if (
            receipt["interpreter"] != sys.executable
            or receipt["python"] != sys.version.split()[0]
            or any(
                importlib.metadata.version(name) != version
                for name, version in receipt["dependencies"].items()
            )
        ):
            mismatches.append("environment_changed")
        for name, key in (
            ("executed.ipynb", "executed_notebook_sha256"),
            ("report.json", "report_sha256"),
            ("report.html", "report_html_sha256"),
        ):
            if _digest(bounded(name)) != receipt[key]:
                mismatches.append("output_changed")
    except Exception:
        mismatches.append("evidence_unavailable_or_changed")
    return {
        "valid": not mismatches,
        "origin": "local",
        "mismatches": sorted(set(mismatches)),
    }
