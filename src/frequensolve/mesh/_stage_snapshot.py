# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Immutable, relocatable inputs for one patch optimization stage."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Sequence

import h5py
import numpy as np

from frequensolve.imaging._artifacts import ControlStateFile
from frequensolve.model.property import rsf_binary_path
from frequensolve.simulation.jobs.remote import _PROJECT_FILE_REFERENCE_KEYS
from frequensolve.simulation.simulation import CustomJSONEncoder

if TYPE_CHECKING:
    from frequensolve.mesh._stage_mesh import PatchStageMesh
    from frequensolve.mesh.patches import PreparedPatchSet
    from frequensolve.simulation.simulation import SeismicSimulation

_SCHEMA = "fs-patch-stage-1"
_PATH_KEYS = _PROJECT_FILE_REFERENCE_KEYS | {"artifact"}


def _digest(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return "sha256:" + sha.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload, cls=CustomJSONEncoder, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _contained(root: Path, name: str) -> Path:
    """Resolve only bundle-relative paths, including after directory relocation."""
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or "\\" in name
        or ":" in name
    ):
        raise ValueError(f"Stage file must be bundle-relative: {name!r}")
    result = root.joinpath(*path.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Stage file escapes its bundle: {name!r}")
    return result


def _check_hdf_dependencies(path: Path) -> None:
    """Refuse HDF indirection that would escape the copied file's byte identity."""
    if not h5py.is_hdf5(path):
        return
    with h5py.File(path, "r") as h5:
        visited: set[int] = set()

        def walk(group: h5py.Group) -> None:
            address = h5py.h5o.get_info(group.id).addr
            if address in visited:
                return
            visited.add(address)
            for name in group:
                link = group.get(name, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    raise ValueError(
                        "Materialize HDF external links before freezing a stage"
                    )
                node = group[name]
                if isinstance(node, h5py.Group):
                    walk(node)
                elif node.is_virtual or node.external:
                    raise ValueError(
                        "Materialize HDF external or virtual datasets before freezing a stage"
                    )

        walk(h5)


class _InputCopier:
    def __init__(self, root: Path, export_root: Path, project: Path) -> None:
        self.root = root
        self.export_root = export_root
        self.project = project
        self.copied: dict[Path, str] = {}
        (root / "inputs").mkdir()

    def pin(self, locator: str | Path) -> str:
        text = str(locator)
        file_name, separator, dataset = text.partition(":")
        source = Path(file_name).expanduser()
        if not source.is_absolute():
            exported = self.export_root / source
            source = exported if exported.exists() else self.project / source
        source = source.resolve(strict=True)
        if source.is_dir() and self.root.resolve().is_relative_to(source):
            raise ValueError(
                "Stage destination cannot be inside a copied input directory"
            )
        if source not in self.copied:
            relative = f"inputs/{len(self.copied):06d}{source.suffix}"
            self.copied[source] = relative
            target = self.root / relative
            if source.is_dir():
                target.mkdir()
                entries = sorted(source.rglob("*"))
                for entry in entries:
                    if entry.is_symlink():
                        raise ValueError(
                            "Stage input directories cannot contain symlinks"
                        )
                    destination = target / entry.relative_to(source)
                    if entry.is_dir():
                        destination.mkdir()
                    else:
                        self._copy_file(entry, destination)
                if entries != sorted(source.rglob("*")):
                    raise ValueError(
                        f"Stage input directory changed while copying: {source}"
                    )
            else:
                self._copy_file(source, target)
        return self.copied[source] + (separator + dataset if separator else "")

    def _copy_file(self, source: Path, target: Path) -> None:
        _check_hdf_dependencies(source)
        before = _digest(source)
        shutil.copyfile(source, target)
        if _digest(target) != before or _digest(source) != before:
            raise ValueError(f"Stage input changed while copying: {source}")
        if source.suffix.lower() == ".rsf":
            sidecar = rsf_binary_path(source)
            if sidecar is None:
                raise ValueError("Stage RSF inputs require a separate binary sidecar")
            pinned = self.root / self.pin(sidecar)
            relative = os.path.relpath(pinned, target.parent)
            header = target.read_text()
            header, count = re.subn(
                r'(?<!\w)in\s*=\s*("[^"]*"|[^\s]+)',
                lambda _: f'in="{relative}"',
                header,
            )
            if not count:
                raise ValueError("RSF sidecar reference could not be pinned")
            target.write_text(header)

    def rewrite(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    self.pin(item)
                    if key in _PATH_KEYS and isinstance(item, (str, Path))
                    else self.rewrite(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.rewrite(item) for item in value]
        return value


@dataclass(frozen=True)
class PatchStageSnapshot:
    """A verified stage bundle; identity is the SHA-256 of its exact manifest bytes."""

    manifest: Path
    identity: str

    @classmethod
    def publish(
        cls,
        directory: Path,
        simulation: SeismicSimulation,
        state: ControlStateFile | str | Path,
        prepared: PreparedPatchSet,
        *,
        name: str,
        frequencies: Sequence[complex] | np.ndarray,
    ) -> PatchStageSnapshot:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("A patch stage requires a nonempty name")
        band = np.asarray(frequencies, dtype=complex)
        if (
            band.ndim != 1
            or not band.size
            or not np.all(np.isfinite(band))
            or np.any(band.real <= 0)
        ):
            raise ValueError(
                "Patch stages require finite frequencies with positive real parts"
            )
        control_state = (
            deepcopy(state)
            if isinstance(state, ControlStateFile)
            else ControlStateFile.read(state)
        )
        directory = Path(directory).expanduser().resolve()
        if directory.exists():
            raise FileExistsError(
                f"Stage already published; reopen its verified manifest: {directory}"
            )
        directory.parent.mkdir(parents=True, exist_ok=True)
        geometry = prepared.geometry
        basis: dict[str, str] = {}
        for patch in geometry["patches"]:
            for coverage in patch.get("material_coverage", ()):
                space, identity = coverage["space"], coverage["basis_identity"]
                if space in basis and basis[space] != identity:
                    raise ValueError(
                        "Patch preparation contains conflicting parent material bases"
                    )
                basis[space] = identity
        with TemporaryDirectory(
            prefix=".patch-stage-", dir=directory.parent
        ) as temporary:
            temporary_root = Path(temporary)
            root, export = temporary_root / "bundle", temporary_root / "export"
            root.mkdir()
            local = deepcopy(simulation)
            context = local.export_context(project_path=export, rel_path="arrays")
            payload = local.to_fs(context)
            payload["project_path"] = "."
            payload = _InputCopier(root, export, Path(simulation.project_path)).rewrite(
                payload
            )
            parent = Path(prepared.jobs[-1]._result_path) / geometry["parent_file"]
            before = _digest(parent)
            shutil.copyfile(parent, root / "parent.gmp")
            if _digest(root / "parent.gmp") != before or _digest(parent) != before:
                raise ValueError("Prepared parent geometry changed while copying")
            geometry["parent_file"] = "parent.gmp"
            _write_json(root / "simulation.json", payload)
            _write_json(root / "geometry.json", geometry)
            _write_json(root / "acquisition.json", prepared.acquisition)
            control_state.write(root / "control_state.h5")
            files = [
                {
                    "file": path.relative_to(root).as_posix(),
                    "sha256": _digest(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(root.rglob("*"))
                if path.is_file()
            ]
            descriptor = {
                "schema": _SCHEMA,
                "name": name,
                "frequencies": [[value.real, value.imag] for value in band],
                "parent_geometry": geometry["parent_fingerprint"],
                "parent_mesh": "parent.gmp",
                "material_basis": basis,
                "pml": prepared.pml.to_fs(),
                "geometry_identity": _digest(root / "geometry.json"),
                "acquisition_identity": _digest(root / "acquisition.json"),
                "simulation": "simulation.json",
                "control_state": "control_state.h5",
                "files": files,
            }
            _write_json(root / "manifest.json", descriptor)
            identity = _digest(root / "manifest.json")
            cls.read(root / "manifest.json", identity=identity)
            if prepared.jobs[-1].geometry_report != prepared.geometry:
                raise ValueError("Patch preparation changed while freezing the stage")
            # The complete nonempty directory appears at once; never replace an existing stage.
            root.rename(directory)
        return cls.read(directory / "manifest.json", identity=identity)

    @classmethod
    def read(cls, manifest: str | Path, *, identity: str) -> PatchStageSnapshot:
        manifest_path = Path(manifest).resolve(strict=True)
        if (
            not re.fullmatch(r"sha256:[0-9a-f]{64}", identity)
            or _digest(manifest_path) != identity
        ):
            raise ValueError("Patch stage manifest identity mismatch")
        payload = json.loads(manifest_path.read_text())
        if payload.get("schema") != _SCHEMA:
            raise ValueError("Unsupported patch stage schema")
        band = np.asarray(payload.get("frequencies", []), dtype=float)
        if (
            band.ndim != 2
            or band.shape[1] != 2
            or not len(band)
            or not np.all(np.isfinite(band))
            or np.any(band[:, 0] <= 0)
        ):
            raise ValueError("Invalid patch stage frequency band")
        root = manifest_path.parent
        files = payload["files"]
        by_name = {}
        for record in files:
            name = record["file"]
            if name in by_name:
                raise ValueError("Duplicate patch stage file")
            by_name[name] = record
            path = _contained(root, name)
            if (
                not path.is_file()
                or path.stat().st_size != record["bytes"]
                or _digest(path) != record["sha256"]
            ):
                raise ValueError(f"Patch stage input changed: {name}")
        for name in (
            payload["simulation"],
            payload["control_state"],
            payload["parent_mesh"],
            "geometry.json",
            "acquisition.json",
        ):
            if name not in by_name:
                raise ValueError(f"Patch stage omits required file: {name}")
        if (
            by_name["geometry.json"]["sha256"] != payload["geometry_identity"]
            or by_name["acquisition.json"]["sha256"] != payload["acquisition_identity"]
        ):
            raise ValueError("Patch stage geometry/acquisition identity mismatch")
        ControlStateFile.read(_contained(root, payload["control_state"]))
        return cls(manifest_path, identity)

    def verify(self) -> PatchStageSnapshot:
        """Revalidate before each use; a saved Python object does not certify mutable disk bytes."""
        return self.read(self.manifest, identity=self.identity)

    def to_fs(self) -> dict[str, str]:
        self.verify()
        return {"manifest": str(self.manifest), "identity": self.identity}

    def input_files(self) -> tuple[Path, ...]:
        """Enumerate verified committed bytes without including generated stage jobs."""
        self.verify()
        payload = json.loads(self.manifest.read_text())
        return (self.manifest,) + tuple(
            _contained(self.manifest.parent, record["file"])
            for record in payload["files"]
        )

    def publish_mesh(
        self, capture_manifest: str | Path, directory: str | Path
    ) -> PatchStageMesh:
        """Publish one completed native frequency-mesh capture beside the stage inputs."""
        from frequensolve.mesh._stage_mesh import PatchStageMesh

        return PatchStageMesh.publish(capture_manifest, directory, stage=self)

    @property
    def control_state(self) -> Path:
        self.verify()
        return (
            self.manifest.parent
            / json.loads(self.manifest.read_text())["control_state"]
        )

    @property
    def simulation_file(self) -> Path:
        self.verify()
        return (
            self.manifest.parent / json.loads(self.manifest.read_text())["simulation"]
        )
