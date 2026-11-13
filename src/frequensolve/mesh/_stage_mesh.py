# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Verified publication of a native stage's realized frequency mesh."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from frequensolve.mesh._stage_snapshot import PatchStageSnapshot, _contained, _digest

_FILES = ("initial.h5", "refinements.h5", "final.h5")


@dataclass(frozen=True)
class PatchStageMesh:
    """Immutable companion to stage inputs, for one patch and physical frequency."""

    manifest: Path
    identity: str

    @classmethod
    def publish(
        cls, source: str | Path, directory: str | Path, *, stage: PatchStageSnapshot
    ) -> PatchStageMesh:
        """Atomically copy a completed native capture without changing its identity."""
        stage.verify()
        source_path = Path(source).resolve(strict=True)
        captured = cls.read(source_path, identity=_digest(source_path), stage=stage)
        destination = Path(directory).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"Stage mesh already published: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".stage-mesh-", dir=destination.parent
        ) as temporary:
            root = Path(temporary) / "bundle"
            root.mkdir()
            for name in ("manifest.json", *_FILES):
                original = (
                    source_path
                    if name == "manifest.json"
                    else source_path.parent / name
                )
                shutil.copyfile(original, root / name)
            captured.verify(stage=stage)
            cls.read(root / "manifest.json", identity=captured.identity, stage=stage)
            root.rename(destination)
        return cls.read(
            destination / "manifest.json", identity=captured.identity, stage=stage
        )

    @classmethod
    def read(
        cls,
        manifest: str | Path,
        *,
        identity: str,
        stage: PatchStageSnapshot | None = None,
    ) -> PatchStageMesh:
        manifest_path = Path(manifest).resolve(strict=True)
        if (
            not re.fullmatch(r"sha256:[0-9a-f]{64}", identity)
            or _digest(manifest_path) != identity
        ):
            raise ValueError("Stage mesh manifest identity mismatch")
        payload = json.loads(manifest_path.read_text())
        if payload.get("schema") != "fs-stage-mesh-1":
            raise ValueError("Unsupported stage mesh schema")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", payload["execution_identity"]):
            raise ValueError("Invalid stage mesh execution identity")
        if set(payload["files"]) != set(_FILES):
            raise ValueError(
                "Stage mesh requires initial, refinement and final artifacts"
            )
        if stage is not None:
            stage.verify()
            if payload["context"]["stage_identity"] != stage.identity:
                raise ValueError("Stage mesh belongs to a different input stage")
        for name in _FILES:
            record = payload["files"][name]
            path = _contained(manifest_path.parent, name)
            if (
                path.stat().st_size != record["bytes"]
                or _digest(path) != record["sha256"]
            ):
                raise ValueError(f"Stage mesh file changed: {name}")
        return cls(manifest_path, identity)

    def verify(self, *, stage: PatchStageSnapshot | None = None) -> PatchStageMesh:
        return self.read(self.manifest, identity=self.identity, stage=stage)

    def input_files(self) -> tuple[Path, ...]:
        """Enumerate the verified manifest and all three mesh replay artifacts."""
        self.verify()
        return (self.manifest,) + tuple(self.manifest.parent / name for name in _FILES)

    def to_fs(self) -> dict[str, str]:
        self.verify()
        return {
            "mode": "replay",
            "manifest": str(self.manifest),
            "identity": self.identity,
        }
