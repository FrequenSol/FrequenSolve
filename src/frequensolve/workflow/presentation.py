"""Read-only scientific chapters for a customer's existing MCP conversation.

No Cloud client, submission, notebook execution, HTML or host-specific UI lives
here. Structured provenance accompanies portable Markdown and bounded PNGs.
"""

from __future__ import annotations

import base64
import io
import math
import threading
from dataclasses import dataclass
from typing import Any, Literal

from mcp.types import ImageContent, TextContent

from .analysis import OPERATION, analyze_pressure
from .storage import canonical
from .study import model_summary

Chapter = Literal["overview", "model", "plan", "progress", "results", "reproduce"]
VERSION = "scientific-conversation.v1"
_PLOT_LOCK = threading.Lock()  # Matplotlib has process-global style/font state.


@dataclass(frozen=True)
class Presentation:
    narrative: str
    provenance: dict
    figures: tuple[bytes, ...] = ()


def text(value: object, maximum: int = 2048) -> str:
    """Escape stored customer labels as Markdown data, never active HTML/links."""
    value = "Unavailable" if value is None else str(value)
    value = " ".join(value.split())[:maximum]
    for character in "\\`*_{}[]<>()|#":
        value = value.replace(character, "\\" + character)
    return value


def number(value: object) -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return "Unavailable"
    try:
        numeric = float(value)
        return f"{numeric:.6g}" if math.isfinite(numeric) else "Unavailable"
    except (TypeError, ValueError):
        return "Unavailable"


def table(headers: tuple[str, ...], rows: list[list[object]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(text(cell, 256) for cell in row) + " |" for row in rows]
    )


def current_reports(view: dict) -> tuple[list[dict], str]:
    """Recalculate plotted values; never silently attach a stale saved analysis.

    A saved comparison remains a recorded comparison. Its reference bytes are
    not downloaded or independently verified again by this read operation.
    """
    plan, study = view["plan"], view.get("study") or {}
    analysis = study.get("analysis")
    if not analysis:
        return [], "No saved analysis is available for this study."
    head = plan["head"]
    bindings = {
        "execution_id": head.get("execution_id"),
        "plan_revision": head["revision"],
        "input_digest": plan.get("input_digest"),
        "plan_digest": (plan.get("plan") or {}).get("digest"),
    }
    if any(
        value is None or analysis.get(key) != value for key, value in bindings.items()
    ):
        return (
            [],
            "The saved analysis belongs to an earlier or different input, plan or execution. Reopen that exact run or analyze the current outputs.",
        )
    reports = analysis.get("reports", [])
    if not isinstance(reports, list) or not 1 <= len(reports) <= 2:
        raise ValueError("Saved analysis exceeds the supported frequency selection")
    checked, indices = [], set()
    for report in reports:
        evidence = report["assurance"]["input"]
        index = evidence.get("frequency_index")
        if (
            report.get("operation") != OPERATION
            or type(index) is not int
            or index not in (0, 1)
            or index in indices
            or evidence.get("plan_id") != head["plan_id"]
            or evidence.get("execution_id") != bindings["execution_id"]
            or evidence.get("plan_revision") != bindings["plan_revision"]
            or not analysis.get("result_digest")
            or evidence.get("result_digest") != analysis["result_digest"]
        ):
            raise ValueError("Saved pressure evidence does not match the current study")
        indices.add(index)
        frequencies = plan.get("scientific_inputs", {}).get("frequencies", [])
        if (
            index >= len(frequencies)
            or len(frequencies[index]) != 2
            or any(
                float(report["metadata"][name]) != float(expected)
                for name, expected in zip(("frequency", "laplace"), frequencies[index])
            )
        ):
            raise ValueError(
                "Saved frequency coordinates do not match the scientific inputs"
            )
        calculated = analyze_pressure(
            {
                "version": "pressure-snapshot.v1",
                "metadata": report["metadata"],
                "real": report["real"],
                "imaginary": report["imaginary"],
                "evidence": evidence,
            }
        )
        derived = (
            "sample_count",
            "finite_count",
            "magnitude",
            "principal_phase_radians",
        )
        matches = all(
            canonical(calculated[key]) == canonical(report.get(key)) for key in derived
        )
        calculated["recorded_comparison"] = (
            report.get("comparison") if matches else None
        )
        calculated["recorded_summary_matches"] = matches
        checked.append(calculated)
    return (
        checked,
        "Saved local analysis; pressure magnitude and phase recalculated for this view. Input files and reference comparisons have not been independently reverified by this read.",
    )


def model_text(science: dict) -> list[str]:
    summary = model_summary(science)
    lines = [
        f"**Model:** {text(science.get('dimension'))}D {text(science.get('physics'))}, {text(science.get('precision'))} precision.",
    ]
    units = science.get("units", {}).get("defaults", {})
    mesh = science.get("mesh", {})
    generator, adapt = mesh.get("generator", {}), mesh.get("adapt", {})
    if generator.get("l_bound") is not None and generator.get("u_bound") is not None:
        lines.append(
            f"**Domain:** {text(generator['l_bound'])} to {text(generator['u_bound'])} {text(units.get('length'))}."
        )
    materials = science.get("model", {}).get("subdomains", [])
    if len(materials) == 1:
        props = materials[0].get("properties", {})
        lines.append(
            f"**Material:** velocity {text(props.get('vp', {}).get('value'))} {text(units.get('velocity'))}; density {text(props.get('rho', {}).get('value'))} {text(units.get('density'))}."
        )
    frequencies = science.get("frequencies", [])
    if frequencies:
        lines.append(
            "**Frequency / Laplace coordinates:** "
            + "; ".join(f"{text(f)} Hz / {text(l)}" for f, l in frequencies)
            + "."
        )
    boundaries = science.get("boundaries", [])
    if boundaries:
        lines.append(
            "**Boundaries:** "
            + "; ".join(
                f"{text(', '.join(b.get('boundaries', [])))}: {text(', '.join(b.get('conditions', [])))}"
                for b in boundaries
            )
            + ". PML denotes the absorbing boundary treatment."
        )
    if adapt:
        lines.append(
            f"**Mesh:** polynomial order {text(adapt.get('order'))}; {text(adapt.get('elems_per_wave'))} elements per wavelength. This setting alone does not establish mesh convergence."
        )
    solver = science.get("solver", {})
    if solver:
        lines.append(
            f"**Solver:** {text(solver.get('coarse_solver'))}; tolerance {text(solver.get('tolerance'))}."
        )
    if summary["wavelengths"]:
        lines.append(
            "**Scale estimate — wavelength = velocity ÷ frequency:** "
            + "; ".join(
                f"{number(row['frequency_hz'])} Hz → {number(row['wavelength_m'])} m"
                for row in summary["wavelengths"]
            )
            + ". This is a homogeneous-medium estimate, not a numerical accuracy test."
        )
    else:
        lines.append(
            "A supported wavelength estimate is unavailable from these inputs."
        )
    return lines


def geometry(science: dict) -> dict | None:
    """Only explicit inline 3D coordinates with matching units; no conversion."""
    import numpy as np

    try:
        units = science["units"]["defaults"]["length"]
        acquisition = science["acquisition"]
        sources = acquisition["source_geometry"]["sources"]
        groups = acquisition["receiver_groups"]
        if science["dimension"] != 3 or units not in ("m", "km"):
            return None
        if not groups or any(g["coordinates"]["units"] != units for g in groups):
            return None
        receivers = [xyz for g in groups for xyz in g["coordinates"]["value"]]
        source = np.asarray([s["coordinates"] for s in sources], dtype=float)
        receiver = np.asarray(receivers, dtype=float)
        bounds = science["mesh"]["generator"]
        low, high = np.asarray(bounds["l_bound"], dtype=float), np.asarray(
            bounds["u_bound"], dtype=float
        )
        if (
            low.shape != (3,)
            or high.shape != (3,)
            or not np.isfinite(low).all()
            or not np.isfinite(high).all()
            or not (high > low).all()
            or any(
                a.ndim != 2
                or a.shape[1] != 3
                or not 1 <= len(a) <= 32
                or not np.isfinite(a).all()
                for a in (source, receiver)
            )
        ):
            return None
        if any(((a < low) | (a > high)).any() for a in (source, receiver)):
            return None
        return {
            "sources": source,
            "receivers": receiver,
            "low": low,
            "high": high,
            "units": units,
        }
    except (KeyError, TypeError, ValueError):
        return None


def figure_png(
    *, model: dict | None = None, reports: list[dict] | None = None
) -> bytes:
    """Discrete pressure plots or an explicit model diagram; no interpolated field."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with _PLOT_LOCK:
        fig = Figure(
            figsize=(10, 4.7 if model else max(3.4, 3.1 * len(reports or []))), dpi=120
        )
        FigureCanvasAgg(fig)
        fig.set_facecolor("#fafcfb")
        if model:
            from itertools import combinations, product

            ax = fig.add_subplot(111, projection="3d")
            corners = np.array(list(product(*zip(model["low"], model["high"]))))
            for a, b in combinations(corners, 2):
                if np.count_nonzero(a != b) == 1:
                    ax.plot(*np.array([a, b]).T, color="#94aba9", linewidth=0.9)
            ax.scatter(
                *model["receivers"].T,
                marker="o",
                facecolors="none",
                edgecolors="#18776c",
                s=180,
                linewidths=2,
                label="Pressure receivers",
            )
            ax.scatter(
                *model["sources"].T, marker="*", color="#ad6414", s=100, label="Sources"
            )
            for i, xyz in enumerate(model["sources"], 1):
                ax.text(*xyz, f"  S{i}", fontsize=9)
            for axis, low, high in zip("xyz", model["low"], model["high"]):
                getattr(ax, f"set_{axis}lim")(low, high)
                getattr(ax, f"set_{axis}label")(
                    f"{axis.upper()} ({model['units']})", labelpad=8
                )
            ax.set_box_aspect(model["high"] - model["low"])
            ax.set_title(
                "The acoustic model · specified geometry",
                fontsize=14,
                loc="left",
                pad=20,
            )
            ax.legend(loc="upper left", frameon=False, fontsize=9)
            fig.subplots_adjust(left=0.02, right=0.94, top=0.88, bottom=0.12)
        else:
            if not reports:
                raise ValueError("Choose model geometry or pressure reports")
            axes = fig.subplots(len(reports), 2, squeeze=False)
            for row, report in enumerate(reports):
                meta = report["metadata"]
                size = report["sample_count"]
                if size > 64:
                    raise ValueError("Select at most 64 pressure samples for a figure")
                labels = [
                    f"S{s}→R{r}"
                    for s in meta["source_ids"]
                    for r in meta["receiver_ids"]
                ]
                for col, (field, title, units) in enumerate(
                    (
                        ("magnitude", "Pressure magnitude", meta["units"]),
                        ("principal_phase_radians", "Principal phase", "rad"),
                    )
                ):
                    ax = axes[row, col]
                    values = np.array(report[field], dtype=float).reshape(-1)
                    ax.scatter(range(size), values, s=48, color="#18776c", zorder=3)
                    ax.set_xticks(
                        range(size), labels, rotation=25, ha="right", fontsize=9
                    )
                    ax.set_title(
                        f"{number(meta['frequency'])} Hz · {title}",
                        loc="left",
                        fontsize=12,
                    )
                    ax.set_ylabel(text(units, 32))
                    ax.grid(axis="y", alpha=0.2)
                    ax.spines[["top", "right"]].set_visible(False)
                    if col == 1:
                        ax.set_ylim(-math.pi * 1.1, math.pi * 1.1)
                        ax.set_yticks([-math.pi, 0, math.pi], ["−π", "0", "π"])
                    if not np.isfinite(values).all():
                        ax.text(
                            0.02,
                            0.98,
                            "Undefined values omitted",
                            transform=ax.transAxes,
                            va="top",
                            fontsize=8,
                        )
            fig.tight_layout(pad=2.2)
        output = io.BytesIO()
        fig.savefig(output, format="png", metadata={"Software": VERSION})
        return output.getvalue()


def present(
    view: dict, chapter: Chapter, indices: list[int], figures: bool = True
) -> Presentation:
    if len(canonical(view)) > 120_000 or chapter not in (
        "overview",
        "model",
        "plan",
        "progress",
        "results",
        "reproduce",
    ):
        raise ValueError("Unsupported bounded study presentation")
    if (
        not indices
        or len(indices) > 2
        or len(set(indices)) != len(indices)
        or any(type(i) is not int or i not in (0, 1) for i in indices)
    ):
        raise ValueError("Select distinct supported frequencies")
    plan, study, credits = view["plan"], view.get("study") or {}, view["credits"]
    head, science = plan["head"], plan.get("scientific_inputs") or {}
    lines, images = [], []
    provenance = {
        "version": VERSION,
        "chapter": chapter,
        "head": head,
        "input_digest": plan.get("input_digest"),
        "plan_digest": (plan.get("plan") or {}).get("digest"),
        "observation": "scoped service snapshot",
        "paid_request_sent": False,
        "service_attestation": False,
        "input_files_reverified": False,
        "next_action": None,
    }
    title = text(study.get("title") or "Your acoustic study", 128)
    if chapter == "overview":
        lines = [
            f"**{title}**",
            f"**Scientific question:** {text(study.get('question'))}",
            f"**Success criterion:** {text(study.get('success_criterion'))}",
            "We will work through the model and estimates, review the compute proposal, follow the run and Credits, then inspect pressure results and reproduce the analysis. You can ask about any step here.",
            f"**Saved workflow state:** {text(head.get('status'))}. This state alone does not establish scientific success.",
        ]
        provenance["next_action"] = "model"
    elif chapter == "model":
        lines = [f"**{title} · model and estimates**", *model_text(science)]
        model = geometry(science)
        if model:
            lines.append(
                f"**Geometry:** {len(model['sources'])} sources and {len(model['receivers'])} pressure receivers. The figure shows the specified domain and coordinates; it is not a simulated wavefield."
            )
            lines.append(
                table(
                    ("Kind", "X", "Y", "Z", "Units"),
                    [
                        [kind, *[number(x) for x in point], model["units"]]
                        for kind, key in (
                            ("Source", "sources"),
                            ("Receiver", "receivers"),
                        )
                        for point in model[key]
                    ],
                )
            )
            if figures:
                try:
                    images.append(figure_png(model=model))
                except ImportError:
                    lines.append(
                        "Figure unavailable in this installation; the coordinate table remains usable."
                    )
        else:
            lines.append(
                "An explicit geometry diagram is unavailable from these inputs; no positions or unit conversions were inferred."
            )
        provenance["next_action"] = "plan"
    elif chapter == "plan":
        lines = [
            f"**{title} · compute proposal**",
            f"Saved plan revision {text(head.get('revision'))}.",
        ]
        rows = (plan.get("plan") or {}).get("rows", [])
        provenance["resource_plan"] = plan.get("plan")
        if rows:
            display = []
            for row in rows:
                requested = row.get("requested") or {}
                layout = requested.get("layout") or {}
                ranks, cpus = layout.get("ranks_per_node"), layout.get("cpus_per_rank")
                display.append(
                    [
                        row.get("frequency"),
                        layout.get("nodes"),
                        (
                            ranks * cpus
                            if type(ranks) is int and type(cpus) is int
                            else None
                        ),
                        requested.get("memory_mib_per_node"),
                        requested.get("wall_time_seconds"),
                        row.get("validation", {}).get("status"),
                    ]
                )
                if row.get("reason"):
                    lines.append(
                        f"**{text(row.get('frequency'))} Hz:** {text(row['reason'])}."
                    )
                for key, new in (row.get("overrides") or {}).items():
                    lines.append(
                        f"Resource edit at {text(row.get('frequency'))} Hz — {text(key.replace('_', ' '))}: {text((row.get('automatic') or {}).get(key))} → {text(new)}."
                    )
            lines.append(
                table(
                    (
                        "Hz",
                        "CPU nodes",
                        "CPUs/node",
                        "MiB/node",
                        "Time limit (s)",
                        "Validation",
                    ),
                    display,
                )
            )
            lines.append(
                "Wall limits are limits, not completion-time estimates. Available capacity and measured distributed peak memory are not supplied by this view."
            )
        else:
            lines.append(
                "Automatic resource choices are not available yet. Review the separate sizing proposal before paid planning."
            )
        lines.append(
            "Planning, newly required paid validation and execution have separate approvals. The approval tool refreshes the exact proposal and Credit maximum; this presentation cannot approve or start work."
        )
        provenance["next_action"] = "review_current_stage_or_observe_existing_operation"
    elif chapter == "progress":
        execution, planning = plan.get("execution") or {}, plan.get("planning") or {}
        try:
            reports, analysis_note = current_reports(view)
        except (KeyError, TypeError, ValueError):
            reports, analysis_note = (
                [],
                "Saved analysis could not be verified; execution and financial observations below remain independent.",
            )
        lines = [
            f"**{title} · progress and Credits**",
            {
                "SUCCEEDED": "The service records successful execution. Inspect the verified outputs before drawing a scientific conclusion.",
                "FAILED": "The execution failed. Inspect its diagnostics before deciding whether to run again; observing this operation does not retry it.",
                "CANCELED": "The execution was canceled. Check retained outputs and settlement separately.",
            }.get(
                str(execution.get("outcome")),
                "The execution outcome is not yet confirmed.",
            ),
            table(
                ("Observation", "State"),
                [
                    ["Plan", head.get("status")],
                    ["Planning phase", planning.get("phase")],
                    ["Execution outcome", execution.get("outcome")],
                    ["Execution settlement phase", execution.get("phase")],
                    ["Cancellation requested", execution.get("cancel_requested")],
                    [
                        "Saved analysis",
                        "Available for current run" if reports else analysis_note,
                    ],
                ],
            ),
            "Current download availability is not checked here. Solver completion, publication and financial settlement are independent.",
        ]
        observations = credits.get("observations", [])
        if observations:
            lines.append(
                table(
                    (
                        "Stage",
                        "Quote maximum",
                        "Authorized maximum",
                        "Captured Credits",
                        "Reconciliation",
                    ),
                    [
                        [
                            r.get("purpose"),
                            r.get("quoted_maximum_credits"),
                            r.get("authorized_maximum_credits"),
                            r.get("captured_credits"),
                            r.get("reconciliation"),
                        ]
                        for r in observations
                    ],
                )
            )
        lines.append(
            f"**Captured Credits across the available history:** {text(credits.get('captured_credits')) if credits.get('history_complete') else 'Unavailable — history is incomplete'}. Settlement pending: {text(credits.get('settlement_pending'))}."
        )
        lines.append(
            "Charges use the service's settlement receipts; Chargebee remains authoritative. Internal reserved exposure is not a Chargebee hold. An account-balance change is not this study's cost."
        )
        provenance["credit_history"] = credits
        provenance["next_action"] = (
            "results"
            if reports
            else "observe_same_operation_and_check_result_publication"
        )
    elif chapter == "results":
        reports, note = current_reports(view)
        selected = [
            r for r in reports if r["assurance"]["input"]["frequency_index"] in indices
        ]
        missing = sorted(
            set(indices)
            - {r["assurance"]["input"]["frequency_index"] for r in selected}
        )
        lines = [f"**{title} · receiver response**", note]
        if missing:
            lines.append(
                "**Missing selected frequency indices:** "
                + ", ".join(map(str, missing))
                + ". No values were filled in."
            )
        rows = []
        for report in selected:
            meta = report["metadata"]
            lines.append(
                f"**{number(meta['frequency'])} Hz:** {report['finite_count']} / {report['sample_count']} finite pressure values; solver convergence is recorded. Units: {text(meta['units'])}."
            )
            comparison = report["recorded_comparison"]
            if not report["recorded_summary_matches"]:
                lines.append(
                    "The recorded derived summary differs from this recalculation; its reference comparison is withheld."
                )
            if comparison:
                lines.append(
                    f"Recorded reference comparison: relative L2 difference {number(comparison.get('relative_l2_difference'))}; complex RMS difference {number(comparison.get('complex_rms_difference'))} {text(meta['units'])}; paired coverage {number(comparison.get('coverage'))}."
                )
            else:
                lines.append(
                    "No compatible reference comparison is recorded; this is not a passed comparison."
                )
            for s, sid in enumerate(meta["source_ids"]):
                for r, rid in enumerate(meta["receiver_ids"]):
                    if len(rows) < 16:
                        rows.append(
                            [
                                number(meta["frequency"]),
                                f"S{sid} → R{rid}",
                                number(report["real"][s][r]),
                                number(report["imaginary"][s][r]),
                                number(report["magnitude"][s][r]),
                                number(report["principal_phase_radians"][s][r]),
                            ]
                        )
        if rows:
            lines.append(
                table(
                    ("Hz", "Pair", "Real", "Imaginary", "Magnitude", "Phase (rad)"),
                    rows,
                )
            )
            if sum(r["sample_count"] for r in selected) > 16:
                lines.append(
                    "The table shows only the first 16 selected values; complete arrays remain in structured data."
                )
            if figures and all(r["sample_count"] <= 64 for r in selected):
                try:
                    images.append(figure_png(reports=selected))
                except ImportError:
                    lines.append(
                        "Figure unavailable in this installation; use the numeric table."
                    )
            elif figures:
                lines.append(
                    "The selection exceeds the 64-sample figure limit; complete arrays remain in structured data."
                )
        lines.append(
            "Phase is principal phase and is undefined at zero magnitude. No interpolation, rescaling, phase unwrapping or time reconstruction is performed. Reference agreement and solver convergence do not establish physical or mesh accuracy."
        )
        provenance.update(
            reports=selected,
            missing_frequency_indices=missing,
            analysis_origin=study.get("analysis_origin"),
            operation=OPERATION,
        )
        provenance["next_action"] = (
            "reproduce" if selected else "retrieve_and_analyze_current_outputs"
        )
    else:
        lines = [
            f"**{title} · reproduce and explore**",
            "Ask to create the editable study notebook from the verified local result bundle and a compatible reference, when available. The notebook contains the question, assumptions, estimates, pressure analysis, figures and evidence.",
            "Run All performs local analysis only. It does not upload, authorize or submit compute. Each successful fresh-kernel replay records the code, dependencies, inputs, parameters and outputs. This view does not assert that a notebook has already replayed successfully.",
            "Edit a copy to explore another interpretation. Changed code, inputs or environment invalidate the original execution evidence. A changed scientific model requires its own preparation and paid-stage review.",
        ]
        provenance["next_action"] = "create_or_verify_analysis_notebook"
    lines.append(
        "**Next:** "
        + {
            "model": "Inspect the model and its scale estimates here.",
            "plan": "Inspect the proposed resources and any edits here.",
            "review_current_stage_or_observe_existing_operation": "Review the current paid stage, or follow work that is already authorized.",
            "results": "Inspect the selected receiver pressures here.",
            "observe_same_operation_and_check_result_publication": "Check the same saved operation and its published results.",
            "reproduce": "Reproduce this analysis in an editable notebook.",
            "retrieve_and_analyze_current_outputs": "Retrieve and verify the current outputs before interpreting them.",
            "create_or_verify_analysis_notebook": "Create a notebook, or verify the existing replay evidence.",
        }[provenance["next_action"]]
    )
    return Presentation("\n\n".join(lines), provenance, tuple(images))


def render_content(reply: Any) -> list[TextContent | ImageContent]:
    if not reply.ok:
        return [
            TextContent(
                type="text",
                text=reply.diagnostic
                or "This observation could not be confirmed. Retry the same read with the saved identities.",
            )
        ]
    return [TextContent(type="text", text=reply.narrative)] + [
        ImageContent(
            type="image",
            mime_type="image/png",
            data=base64.b64encode(data).decode("ascii"),
        )
        for data in reply._figures
    ]
