"""Restricted control client and local curated input preparation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import urlsplit
from urllib.request import Request, build_opener

from frequensolve.project_identity import ProjectIdentity

from . import CONTRACT
from .auth import WorkflowAuth, WorkflowError, _NoRedirect, exchange
from .journal import IntentJournal
from .storage import PrivateStore, canonical, directory

MUTATIONS = frozenset(
    {"bind_project", "upload_begin", "plan_create", "plan_revise", "plan_validate"}
)
OPERATIONS = MUTATIONS | {
    "discover",
    "upload_ticket",
    "upload_finalize",
    "plan_get",
    "plan_list",
    "plan_quote",
    "plan_results",
    "plan_files",
    "result_ticket",
    "credit_history",
}


def relative(value: str) -> str:
    if (
        not value
        or len(value) > 512
        or re.search(r"[\\:\x00-\x1f\x7f]", value)
        or PurePosixPath(value).is_absolute()
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise ValueError("Use a safe relative path beneath the configured root")
    return value


def read_local(root: Path, name: str, maximum: int = 16 * 1024 * 1024) -> bytes:
    parts = relative(name).split("/")
    with directory(root.joinpath(*parts[:-1])) as fd:
        stream_fd = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
        )
        with os.fdopen(stream_fd, "rb") as stream:
            import stat

            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
                raise ValueError("Local input is not a bounded regular file")
            value = stream.read(maximum + 1)
            if len(value) > maximum:
                raise ValueError("Local input exceeds the limit")
            return value


def write_local(root: Path, name: str, value: bytes) -> None:
    parts = relative(name).split("/")
    with directory(root.joinpath(*parts[:-1]), create=True) as fd:
        target = os.open(
            parts[-1],
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
            dir_fd=fd,
        )
        with os.fdopen(target, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(fd)


def create_project(root: Path, name: str, display_name: str) -> dict:
    name = relative(name)
    if (
        len(display_name) > 128
        or not display_name.strip()
        or re.search(r"[\x00-\x1f\x7f]", display_name)
    ):
        raise ValueError("Use a short project display name")
    parent, leaf = Path(name).parent, Path(name).name
    with directory(root / parent) as fd:
        os.mkdir(leaf, mode=0o700, dir_fd=fd)
    project = root / name
    identity = ProjectIdentity.new()
    write_local(
        project, ".frequensolve/project-identity.json", canonical(identity.to_dict())
    )
    write_local(
        project,
        ".frequensolve/workflow-project.json",
        canonical(
            {"template": "qualified-acoustic-3d-2-20.v1", "display_name": display_name}
        ),
    )
    for template, destination in [
        ("acoustic_3d.json", "simulations/acoustic_3d/acoustic_3d.json"),
        ("frequency.json", "jobs/acoustic_3d/frequency/frequency.json"),
    ]:
        value = json.loads(
            Path(__file__).with_name("templates").joinpath(template).read_bytes()
        )
        value["project_path"] = str(project.absolute())
        write_local(project, destination, canonical(value))
    return {
        "project_id": identity.project_id,
        "project_key": identity.storage_prefix,
        "relative_path": name,
        "template": "qualified-acoustic-3d-2-20.v1",
        "scientific_model_editing": False,
    }


def staged_inputs(project: Path) -> tuple[ProjectIdentity, dict[str, bytes], str]:
    """Use SDK serialization in an isolated temporary project, with no Site or login."""
    from frequensolve import load

    identity = ProjectIdentity.from_dict(
        json.loads(read_local(project, ".frequensolve/project-identity.json"))
    )
    with tempfile.TemporaryDirectory(prefix="fs-workflow-prepare-") as temp:
        stage = Path(temp).resolve()
        for name in (
            "simulations/acoustic_3d/acoustic_3d.json",
            "jobs/acoustic_3d/frequency/frequency.json",
        ):
            value = json.loads(read_local(project, name))
            template = json.loads(
                Path(__file__)
                .with_name("templates")
                .joinpath(Path(name).name)
                .read_bytes()
            )
            value.pop("project_path", None)
            template.pop("project_path", None)
            if canonical(value) != canonical(template):
                raise ValueError(
                    "This pilot only supports the unchanged qualified 3D scientific model"
                )
            value["project_path"] = str(stage)
            write_local(stage, name, canonical(value))
        job = load(stage / "jobs/acoustic_3d/frequency/frequency.json")
        job_file, remote_job = job.save_for_remote(
            "AWSSite", identity.storage_prefix, include_job_id=False
        )
        simulation_file, remote_simulation = job.save_simulation_for_remote(
            "AWSSite", identity.storage_prefix
        )
        prefix = identity.storage_prefix + "/"
        if not str(remote_job).startswith(prefix) or not str(
            remote_simulation
        ).startswith(prefix):
            raise WorkflowError("SDK serialization returned a different project")
        return (
            identity,
            {
                str(remote_job)[len(prefix) :]: Path(job_file).read_bytes(),
                str(remote_simulation)[len(prefix) :]: Path(
                    simulation_file
                ).read_bytes(),
            },
            str(remote_job)[len(prefix) :],
        )


def transfer_bytes(ticket: dict, *, data: bytes | None = None) -> bytes:
    """Exact AWS destination only; never follow a ticket redirect or reflect its URL."""
    url = ticket.get("url", "")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
        or not parsed.hostname
        or not re.fullmatch(
            r"[a-z0-9.-]+\.s3\.us-east-1\.amazonaws\.com", parsed.hostname
        )
    ):
        raise WorkflowError(
            "Transfer destination is not the configured Sandbox storage provider"
        )
    size = ticket.get("size")
    if type(size) is not int or not 0 <= size <= 64 * 1024 * 1024:
        raise WorkflowError("Transfer exceeds its file limit")
    digest = ticket.get("sha256") or str(ticket.get("digest", "")).removeprefix(
        "sha256:"
    )
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise WorkflowError("Transfer has no exact content digest")
    if data is not None and (
        len(data) != size or hashlib.sha256(data).hexdigest() != digest
    ):
        raise WorkflowError("Upload input changed")
    try:
        with build_opener(_NoRedirect()).open(
            Request(
                url,
                data=data,
                method="PUT" if data is not None else "GET",
                headers=ticket.get("headers", {}) if data is not None else {},
            ),
            timeout=25,
        ) as response:
            result = response.read(size + 1) if data is None else b""
        if data is None and (
            len(result) != size or hashlib.sha256(result).hexdigest() != digest
        ):
            raise ValueError()
        return result
    except Exception:
        raise WorkflowError(
            "Transfer could not be verified; retain the same session and file identity"
        ) from None


class WorkflowClient:
    def __init__(self, auth: WorkflowAuth, *, request: Callable = exchange) -> None:
        self.auth, self.request = auth, request

    def call(self, operation: str, arguments: dict) -> dict:
        if operation not in OPERATIONS:
            raise WorkflowError("Unsupported restricted workflow operation")
        credentials = self.auth.credentials()
        response = self.request(
            self.auth.connection.issuer.rstrip("/") + "/workflow",
            {"version": CONTRACT, "operation": operation, "arguments": arguments},
            token=credentials["access_token"],
        )
        if (
            response.get("version") != CONTRACT
            or response.get("ok") is not True
            or not isinstance(response.get("value"), dict)
        ):
            raise WorkflowError(
                "Cloud could not confirm the request; recover the same saved intent"
            )
        return response["value"]

    def journal(self, project: Path) -> IntentJournal:
        credentials = self.auth.credentials()
        identity = ProjectIdentity.from_dict(
            json.loads(read_local(project, ".frequensolve/project-identity.json"))
        )
        return IntentJournal(
            PrivateStore(project / ".frequensolve" / "workflow"),
            {
                **self.auth.connection.binding,
                "subject": credentials["subject"],
                "account_id": credentials["account_id"],
                "project_id": identity.project_id,
            },
        )

    def prepare(self, project: Path) -> dict:
        identity, files, job_path = staged_inputs(project)
        journal = self.journal(project)
        presentation = json.loads(
            read_local(project, ".frequensolve/workflow-project.json")
        )
        journal.perform(
            "bind-project",
            "bind_project",
            {
                "manifest": identity.to_dict(),
                "display_name": presentation["display_name"],
            },
            self.call,
        )
        manifest = [
            {
                "path": name,
                "size": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
            for name, value in sorted(files.items())
        ]
        session = journal.perform(
            "upload-inputs",
            "upload_begin",
            {"project_id": identity.project_id, "files": manifest},
            self.call,
        )
        for index, item in enumerate(session["files"]):
            ticket = self.call(
                "upload_ticket",
                {"session_id": session["session_id"], "file_index": index},
            )
            if not ticket.get("complete"):
                transfer_bytes(ticket, data=files[item["path"]])
        finalized = self.call("upload_finalize", {"session_id": session["session_id"]})
        return {**finalized, "job_path": job_path}
