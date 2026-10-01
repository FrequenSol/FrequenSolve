# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Bounded, crash-consistent restart arrays beside one FWI checkpoint.

An FWI checkpoint is a small, atomically replaced HDF5 file whose metadata is
JSON. Control-sized restart arrays never enter that JSON; they live in the
sibling directory ``<checkpoint stem>.restart/``, one subdirectory per stage
run:

``stage.h5``
    Written once per stage: the optimizer's coordinate scaling and the BFGS
    archive's base inverse diagonal, seed modes and eigenvalues, identities
    and provenance.
``<prefix>pair_<id>.h5``
    One file per accepted L-BFGS curvature pair (``step``, ``difference``),
    written once when the pair is accepted and shared by the optimizer restart
    and the curvature archive.
``<prefix>model_<iteration>.h5`` / ``<prefix>scaling_<n>.h5``
    An optimizer-coordinate iterate that differs from the checkpoint model,
    and the coordinate scaling of a local patch solve.

Files are published by atomic rename and never modified afterwards. A
checkpoint records its stage directory and every file it needs; once the
checkpoint itself has been replaced, :meth:`RestartStore.commit` deletes all
other files and stage directories. Crash leftovers (unreferenced files or
temporaries) are removed by the next commit, so the store holds at most the
optimizer's memory plus one new generation, regardless of run length.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set, Tuple

import h5py
import numpy as np

from frequensolve.inversion.optimization import LBFGS_RESTART_SCHEMA, LBFGSRestart
from frequensolve.util.atomic import atomic_output_path

RESTART_SCHEMA = "fs-fwi-restart-1"


def _frozen(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


class StageFiles:
    """One stage run's directory of immutable restart files."""

    def __init__(self, directory: Path, known: Set[str]) -> None:
        self.directory = directory
        self.name = directory.name
        self._known = known

    def put(
        self,
        name: str,
        arrays: Mapping[str, Optional[np.ndarray]],
        attrs: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Publish ``name`` once; files this run already wrote or adopted are kept."""
        if name not in self._known:
            self.directory.mkdir(parents=True, exist_ok=True)
            with atomic_output_path(self.directory / name) as temporary:
                with h5py.File(temporary, "w") as h5:
                    for key, value in arrays.items():
                        if value is not None:
                            h5.create_dataset(key, data=value, dtype=np.float64)
                    h5.attrs["metadata"] = json.dumps(dict(attrs or {}), sort_keys=True)
            self._known.add(name)
        return name

    def get(self, name: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Read one restart file into read-only arrays and its metadata."""
        path = self.directory / name
        if not path.is_file():
            raise FileNotFoundError(f"FWI restart file {path} is missing")
        with h5py.File(path, "r") as h5:
            arrays = {
                key: _frozen(np.asarray(h5[key][()], dtype=np.float64)) for key in h5
            }
            attrs = json.loads(h5.attrs.get("metadata", "{}"))
        return arrays, attrs

    def save_state(
        self, state: LBFGSRestart, prefix: str = "", *, model: bool = True
    ) -> Dict[str, Any]:
        """Persist new pairs (and the iterate) of ``state``; return its JSON record.

        ``model=False`` records that the iterate equals the checkpoint model.
        """
        for identifier, step, difference in zip(
            state.pair_ids, state.steps, state.gradient_differences
        ):
            self.put(
                f"{prefix}pair_{identifier}.h5", dict(step=step, difference=difference)
            )
        model_name = None
        if model:
            model_name = self.put(
                f"{prefix}model_{state.accepted_iterations}.h5", dict(model=state.model)
            )
        return dict(
            schema=LBFGS_RESTART_SCHEMA,
            prefix=prefix,
            pairs=list(state.pair_ids),
            model=model_name,
            history_size=state.history_size,
            accepted_iterations=state.accepted_iterations,
            initial_objective=state.initial_objective,
            improvement=state.improvement,
        )

    def load_state(
        self, record: Mapping[str, Any], model: Optional[np.ndarray] = None
    ) -> LBFGSRestart:
        """Rebuild an :class:`LBFGSRestart` from its record (shared read-only arrays)."""
        if record.get("schema") != LBFGS_RESTART_SCHEMA:
            raise ValueError("Unsupported FWI optimizer restart record")
        prefix = record["prefix"]
        steps, differences = [], []
        for identifier in record["pairs"]:
            arrays, _ = self.get(f"{prefix}pair_{identifier}.h5")
            steps.append(arrays["step"])
            differences.append(arrays["difference"])
        if record["model"] is not None:
            model = self.get(record["model"])[0]["model"]
        if model is None:
            raise ValueError("FWI optimizer restart record needs the checkpoint model")
        if model.flags.writeable:
            model = _frozen(np.array(model, dtype=np.float64, copy=True))
        return LBFGSRestart(
            model=model,
            steps=tuple(steps),
            gradient_differences=tuple(differences),
            pair_ids=tuple(record["pairs"]),
            history_size=record["history_size"],
            accepted_iterations=record["accepted_iterations"],
            initial_objective=record["initial_objective"],
            improvement=record["improvement"],
        )


def state_files(record: Optional[Mapping[str, Any]]) -> Set[str]:
    """Return the file names an optimizer restart record references."""
    if not record:
        return set()
    names = {
        f"{record['prefix']}pair_{identifier}.h5" for identifier in record["pairs"]
    }
    if record.get("model"):
        names.add(record["model"])
    return names


class RestartStore:
    """The ``<checkpoint stem>.restart`` directory of one FWI checkpoint."""

    def __init__(self, checkpoint: Path) -> None:
        self.root = checkpoint.with_name(f"{checkpoint.stem}.restart")
        self._known: Dict[str, Set[str]] = {}

    def new_stage(self, index: int) -> StageFiles:
        """Return a fresh directory for one run of stage ``index``."""
        directory = self.root / f"stage_{int(index)}_{uuid.uuid4().hex[:12]}"
        return StageFiles(directory, self._known.setdefault(directory.name, set()))

    def open_stage(self, record: Mapping[str, Any]) -> StageFiles:
        """Reopen the stage directory a committed checkpoint references."""
        if record.get("schema") != RESTART_SCHEMA:
            raise ValueError("Unsupported FWI restart record")
        name = str(record["directory"])
        if Path(name).name != name or not name.startswith("stage_"):
            raise ValueError("FWI restart record names an invalid directory")
        directory = self.root / name
        # A record without files (e.g. no accepted pair yet) has no directory.
        if record["files"] and not directory.is_dir():
            raise FileNotFoundError(f"FWI restart directory {directory} is missing")
        # Committed files are complete; anything else there is a crash leftover.
        known = self._known.setdefault(name, set())
        known.update(str(entry) for entry in record["files"])
        return StageFiles(directory, known)

    def commit(self, record: Optional[Mapping[str, Any]]) -> None:
        """Delete every restart file the just-saved checkpoint does not reference."""
        if not self.root.is_dir():
            return
        keep_directory = None if record is None else str(record["directory"])
        keep_files = set() if record is None else {str(n) for n in record["files"]}
        for entry in self.root.iterdir():
            if entry.name != keep_directory:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                self._known.pop(entry.name, None)
                continue
            known = self._known.get(entry.name)
            for item in entry.iterdir():
                if item.name not in keep_files:
                    item.unlink(missing_ok=True)
                    if known is not None:
                        known.discard(item.name)

    def files(self) -> Set[Path]:
        """Return every file currently in the store (for diagnostics and tests)."""
        if not self.root.is_dir():
            return set()
        return {path for path in self.root.rglob("*") if path.is_file()}
