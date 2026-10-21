"""Bounded local navigation; saved identities are hints, never Cloud authority."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote

from frequensolve.project_identity import ProjectIdentity

from .client import read_local, relative
from .presentation import number, table, text
from .storage import PrivateStore, directory

MAX_ENTRIES = 1000
PAGE_SIZE = 25


def projects(root: Path, root_id: str, folder: str | None, after: str | None) -> dict:
    """Inspect one explicit folder, never recurse or follow filesystem links."""
    parent = root / relative(folder) if folder else root
    if after is not None and ("/" in relative(after)):
        raise ValueError("Use the returned single-folder cursor")
    with directory(parent) as fd, os.scandir(fd) as entries:
        names: list[str] = []
        for count, entry in enumerate(entries):
            if count >= MAX_ENTRIES:
                raise ValueError("Folder exceeds the navigation limit")
            if entry.name.startswith(".") or not entry.is_dir(follow_symlinks=False):
                continue
            try:
                relative(entry.name)
            except ValueError:
                continue
            names.append(entry.name)
    names = sorted(name for name in names if after is None or name > after)
    found, folders, unreadable = [], [], 0
    for name in names[:PAGE_SIZE]:
        path = f"{folder}/{name}" if folder else name
        try:
            metadata_bytes = read_local(
                root / path, ".frequensolve/workflow-project.json", 4096
            )
        except FileNotFoundError:
            folders.append(path)
            continue
        except (OSError, ValueError):
            unreadable += 1
            continue
        try:
            metadata = json.loads(metadata_bytes)
            identity = ProjectIdentity.from_dict(
                json.loads(
                    read_local(root / path, ".frequensolve/project-identity.json", 4096)
                )
            )
            label = metadata["display_name"]
            if (
                metadata.get("template") != "qualified-acoustic-3d-2-20.v1"
                or not isinstance(label, str)
                or not label.strip()
                or len(label) > 128
            ):
                raise ValueError("Invalid project metadata")
        except (OSError, ValueError, KeyError, TypeError):
            unreadable += 1
        else:
            found.append(
                {
                    "display_name": label,
                    "root_id": root_id,
                    "local_project": path,
                    "project_id": identity.project_id,
                    "project_key": identity.storage_prefix,
                }
            )
    return {
        "projects": found,
        "folders": folders,
        "root_id": root_id,
        "folder": folder,
        "after": after,
        "entries_checked": min(PAGE_SIZE, len(names)),
        "search_scope": "direct_local_subfolders",
        "next_after": names[PAGE_SIZE - 1] if len(names) > PAGE_SIZE else None,
        "unreadable_entries": unreadable,
        "cloud_ownership_checked": False,
        "paid_request_sent": False,
        "pagination": "Folder contents may change between pages; deduplicate by project identity and local path.",
    }


def remember_notebook(
    root: Path,
    project: Path,
    destination: str,
    status: str,
    *,
    binding: dict | None = None,
    label: str | None = None,
) -> str:
    """Bookmark the latest notebook without replacing any notebook content."""
    target = root / relative(destination)
    if binding is not None:
        from .notebook_binding import validate

        validate(binding)
    files = []
    total = 0
    with directory(target) as fd, os.scandir(fd) as entries:
        names: list[str] = []
        for entry in entries:
            if len(names) >= 32:
                raise ValueError("Notebook has too many files")
            names.append(entry.name)
    for name in sorted(names):
        data = read_local(target, name)
        total += len(data)
        if total > 64 * 1024 * 1024:
            raise ValueError("Notebook exceeds the navigation limit")
        files.append(
            {
                "name": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    from .notebook_catalog import archive, bookmark_id, validate_record

    saved = validate_record(
        {
            "destination": destination,
            "recorded_replay_status": status,
            "files": files,
            **({"binding": binding} if binding is not None else {}),
            **({"label": label} if label is not None else {}),
        }
    )
    store = PrivateStore(project / ".frequensolve" / "workflow")
    with store.locked("latest-notebook") as record:
        archive(store, record.read())
        archive(store, saved)
        record.write(saved)
    return bookmark_id(saved)


def notebook_artifacts(root: Path, project: Path, selected: str | None = None) -> dict:
    """Check saved bytes, without executing a kernel or claiming current environment proof."""
    from .notebook_catalog import bookmark_id, get, validate_record

    if selected is not None:
        saved = get(project, selected)["bookmark"]
    else:
        try:
            saved = validate_record(
                json.loads(
                    read_local(
                        project, ".frequensolve/workflow/latest-notebook.json", 16384
                    )
                )
            )
        except FileNotFoundError:
            return {
                "artifacts": [],
                "recorded_replay_status": "unavailable",
                "paid_request_sent": False,
            }
    destination = relative(saved["destination"])
    files = saved["files"]
    if not isinstance(files, list) or not 1 <= len(files) <= 32:
        raise ValueError("Invalid notebook bookmark")
    unchanged, total, links = True, 0, []
    for item in files:
        name = relative(item["name"])
        if "/" in name:
            raise ValueError("Invalid notebook file")
        try:
            data = read_local(root, destination + "/" + name)
        except (OSError, ValueError):
            unchanged = False
        else:
            total += len(data)
            if total > 64 * 1024 * 1024:
                raise ValueError("Notebook exceeds the navigation limit")
            unchanged &= (
                len(data) == item["bytes"]
                and hashlib.sha256(data).hexdigest() == item["sha256"]
            )
        if name in ("analysis.ipynb", "executed.ipynb", "report.html"):
            # Only verified, unchanged bookmarks receive clickable file links.
            links.append({"name": name, "path": str(root / destination / name)})
    binding = saved.get("binding")
    if binding is not None:
        from .notebook_binding import validate

        try:
            validate(binding)
        except (ValueError, TypeError):
            binding = None
    return {
        "notebook_id": bookmark_id(saved),
        "artifacts": links if unchanged else [],
        "recorded_replay_status": saved["recorded_replay_status"],
        "binding": binding,
        "saved_bytes_unchanged": bool(unchanged),
        "current_environment_checked": False,
        "service_attestation": False,
        "paid_request_sent": False,
    }


def notebook_history(project: Path, after: str | None = None) -> dict:
    from .notebook_catalog import page, summary

    result = page(project, after)
    return {
        **{k: v for k, v in result.items() if k != "records"},
        "notebooks": [summary(entry) for entry in result["records"]],
        "paid_request_sent": False,
        "current_environment_checked": False,
        "service_attestation": False,
    }


def narrative(value: dict) -> str:
    if "notebook_attempts" in value or "notebook_attempt" in value:
        from .notebook_navigation import narrative as attempt_narrative

        return attempt_narrative(value)
    if "roots" in value:
        return (
            "Choose a scientific workspace: "
            + ", ".join(text(r) for r in value["roots"])
            + "."
        )
    if "notebooks" in value:
        lines = ["Your registered analysis notebooks"]
        if value["notebooks"]:
            lines.append(
                table(
                    (
                        "Choice",
                        "Notebook",
                        "Folder",
                        "Selected frequencies",
                        "Recorded replay",
                    ),
                    [
                        [
                            i,
                            item["label"],
                            item["folder"],
                            item["frequency_label"],
                            item["recorded_replay_status"],
                        ]
                        for i, item in enumerate(value["notebooks"], 1)
                    ],
                )
            )
        else:
            lines.append("No registered notebooks were confirmed on this page.")
        if value["next_after"]:
            lines.append(
                "More registered entries remain. Continue to the next page and retain earlier choices."
            )
        elif value.get("after") is not None:
            lines.append(
                "This is the final page; retain choices found on earlier pages."
            )
        if value["unreadable_entries"]:
            lines.append(
                "Some registered entries could not be checked; the history remains unconfirmed."
            )
        lines.append(
            "This catalog covers local registrations only; unregistered folders have not been searched. Choose a notebook to check its files. Recorded replay state is not a new environment check or service attestation."
        )
        return "\n\n".join(lines)
    if "projects" in value:
        lines = ["Your local scientific projects on this page", ""]
        lines.extend("- " + text(p["display_name"]) for p in value["projects"])
        if not value["projects"]:
            if value["next_after"]:
                lines.append("No curated projects were found on this page.")
            elif value.get("after") is not None:
                lines.append("No curated projects were found on this final page.")
            elif value["unreadable_entries"]:
                lines.append(
                    "No curated projects could be identified from the readable entries in this local folder."
                )
            elif value["folders"]:
                lines.append(
                    "No curated projects were identified directly in this local folder."
                )
            else:
                lines.append(
                    "No curated projects or inspectable subfolders were found directly in this local folder."
                )
        if value["folders"]:
            lines.append("\nChoose a folder to inspect explicitly:")
            lines.extend("- " + text(folder) for folder in value["folders"])
        if value["next_after"]:
            lines.append(
                "\nMore entries remain. Continue to the next page and keep earlier results."
            )
        elif value.get("after") is not None:
            lines.append(
                "\nThis is the final page. Keep projects found on earlier pages."
            )
        if value["unreadable_entries"]:
            lines.append(
                "Some local project metadata is missing, unreadable or unsupported; those entries remain unconfirmed."
            )
        lines.append(
            "\nThis view covers only direct local subfolders. Other folders and saved Cloud work have not been searched."
        )
        if value["projects"]:
            lines.append(
                "Open a project's owned saved plans to resume it. Cloud checks access on those reads."
            )
        else:
            lines.append(
                "To reopen earlier work, choose its existing parent folder within an allowed workspace, or restore its local project files and retry discovery."
            )
        return "\n".join(lines)
    lines = ["Your saved analysis notebook", ""]
    for item in value["artifacts"]:
        labels = {
            "analysis.ipynb": "Editable notebook",
            "executed.ipynb": "Notebook with recorded outputs",
            "report.html": "Static report",
        }
        lines.append(f"- [{labels[item['name']]}]({quote(item['path'], safe='/')})")
    if value.get("saved_bytes_unchanged"):
        lines.append(
            "\nRecorded replay: "
            + text(value["recorded_replay_status"])
            + ". Saved files are unchanged; the current environment has not been checked. Local evidence is not service attestation."
        )
        if value.get("binding"):
            binding = value["binding"]
            lines.append(
                "Recorded notebook context: plan revision "
                + text(binding.get("plan_revision"))
                + "; selected frequencies "
                + ", ".join(
                    number(p[0]) + " Hz" for p in binding["frequency_coordinates"]
                )
                + ". This project-only view has not compared it with a current study."
            )
        else:
            lines.append(
                "This older bookmark has no exact study/run binding. It is recoverable, but is not verified as the current study notebook."
            )
    elif value.get("recorded_replay_status") == "unavailable":
        lines.append(
            "No notebook bookmark is saved for this project yet. Create an analysis notebook from verified results."
        )
    else:
        lines.append(
            "The saved notebook files have changed or are missing. Replay and verify them before relying on earlier execution evidence."
        )
    return "\n".join(lines)
