# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Verified publication of a native stage's realized frequency mesh."""

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from frequensolve.mesh._stage_snapshot import _contained, _digest

_FILES = ("initial.h5", "refinements.h5", "final.h5")


@dataclass(frozen=True)
class PatchStageMesh:
    """Immutable companion to stage inputs, for one patch and physical frequency."""

    manifest: Path
    identity: str

    @classmethod
    def publish(cls, source, directory, *, stage):
        """Atomically copy a completed native capture without changing its identity."""
        stage.verify()
        source = Path(source).resolve(strict=True)
        captured = cls.read(source, identity=_digest(source), stage=stage)
        directory = Path(directory).expanduser().resolve()
        if directory.exists():
            raise FileExistsError(f"Stage mesh already published: {directory}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".stage-mesh-", dir=directory.parent
        ) as temporary:
            root = Path(temporary) / "bundle"
            root.mkdir()
            for name in ("manifest.json", *_FILES):
                original = source if name == "manifest.json" else source.parent / name
                shutil.copyfile(original, root / name)
            captured.verify(stage=stage)
            cls.read(root / "manifest.json", identity=captured.identity, stage=stage)
            root.rename(directory)
        return cls.read(
            directory / "manifest.json", identity=captured.identity, stage=stage
        )

    @classmethod
    def read(cls, manifest, *, identity, stage=None):
        manifest = Path(manifest).resolve(strict=True)
        if (
            not re.fullmatch(r"sha256:[0-9a-f]{64}", identity)
            or _digest(manifest) != identity
        ):
            raise ValueError("Stage mesh manifest identity mismatch")
        payload = json.loads(manifest.read_text())
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
            path = _contained(manifest.parent, name)
            if (
                path.stat().st_size != record["bytes"]
                or _digest(path) != record["sha256"]
            ):
                raise ValueError(f"Stage mesh file changed: {name}")
        return cls(manifest, identity)

    def verify(self, *, stage=None):
        return self.read(self.manifest, identity=self.identity, stage=stage)

    def input_files(self):
        """Enumerate the verified manifest and all three mesh replay artifacts."""
        self.verify()
        return (self.manifest,) + tuple(self.manifest.parent / name for name in _FILES)

    def to_fs(self):
        self.verify()
        return {
            "mode": "replay",
            "manifest": str(self.manifest),
            "identity": self.identity,
        }
