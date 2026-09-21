"""Restartable optimization history, checkpoint, and result records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import h5py
import numpy as np

__all__ = [
    "LossTerms",
    "OptimizationCheckpoint",
    "OptimizationHistory",
    "OptimizationRecord",
    "OptimizationResult",
]

_HISTORY_SCHEMA = "fs-optimization-history-1"
_CHECKPOINT_SCHEMA = "fs-optimization-checkpoint-1"
_RESULT_SCHEMA = "fs-optimization-result-1"
_FINAL_STATUSES = {"converged", "failed", "stopped"}


def _utc_now() -> str:
    """Return a stable UTC timestamp for serialized run records."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _real_model(value: Any, *, name: str = "model") -> np.ndarray:
    """Return one finite float64 model vector without complex truncation."""

    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, copy=True)


def _model_digest(model: np.ndarray) -> str:
    """Hash a model vector for compact provenance without storing it per record."""

    little_endian = np.asarray(model, dtype="<f8")
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def _finite_optional(value: Optional[float], name: str) -> Optional[float]:
    """Normalize one optional finite scalar."""

    if value is None:
        return None
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _scalar_metrics(values: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalize JSON-safe scalar metrics used for lightweight diagnostics."""

    out: Dict[str, Any] = {}
    for key, value in (values or {}).items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bool) or value is None or isinstance(value, str):
            out[str(key)] = value
        elif isinstance(value, int):
            out[str(key)] = int(value)
        elif isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError(f"optimization metric {key!r} must be finite")
            out[str(key)] = float(value)
        else:
            raise TypeError(f"optimization metric {key!r} must be a scalar")
    return out


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    """Replace a JSON record atomically after writing it in the same directory."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


@dataclass(frozen=True)
class LossTerms:
    """Data and regularization contributions to one least-squares loss."""

    data: float
    regularization: float = 0.0

    def __post_init__(self) -> None:
        data = float(self.data)
        regularization = float(self.regularization)
        if not np.isfinite(data) or data < 0.0:
            raise ValueError("data loss must be finite and nonnegative")
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization loss must be finite and nonnegative")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "regularization", regularization)

    @property
    def total(self) -> float:
        """Return the complete objective value."""

        return self.data + self.regularization

    def to_fs(self) -> Dict[str, float]:
        """Serialize loss terms with an explicit total."""

        return {
            "total": self.total,
            "data": self.data,
            "regularization": self.regularization,
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "LossTerms":
        """Deserialize loss terms and verify any recorded total."""

        value = cls(
            data=float(data["data"]),
            regularization=float(data.get("regularization", 0.0)),
        )
        if "total" in data and not np.isclose(
            float(data["total"]), value.total, rtol=1.0e-12, atol=0.0
        ):
            raise ValueError("serialized optimization loss total is inconsistent")
        return value


@dataclass(frozen=True)
class OptimizationRecord:
    """One objective evaluation or accepted optimizer iteration."""

    index: int
    kind: str
    evaluation: int
    iteration: Optional[int]
    loss: LossTerms
    model_digest: str
    elapsed_seconds: float
    gradient_norm: Optional[float] = None
    step_norm: Optional[float] = None
    step_length: Optional[float] = None
    accepted: Optional[bool] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"evaluation", "iteration"}:
            raise ValueError("optimization record kind must be evaluation or iteration")
        if int(self.index) < 0 or int(self.evaluation) < 0:
            raise ValueError("optimization record counters must be nonnegative")
        if self.iteration is not None and int(self.iteration) < 0:
            raise ValueError("optimization iteration must be nonnegative")
        if not isinstance(self.loss, LossTerms):
            raise TypeError("optimization record loss must be LossTerms")
        if len(str(self.model_digest)) != 64:
            raise ValueError("optimization model digest must be SHA-256")
        elapsed = float(self.elapsed_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("optimization elapsed time must be finite and nonnegative")
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "evaluation", int(self.evaluation))
        object.__setattr__(
            self, "iteration", None if self.iteration is None else int(self.iteration)
        )
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(
            self, "gradient_norm", _finite_optional(self.gradient_norm, "gradient norm")
        )
        object.__setattr__(
            self, "step_norm", _finite_optional(self.step_norm, "step norm")
        )
        object.__setattr__(
            self, "step_length", _finite_optional(self.step_length, "step length")
        )
        object.__setattr__(
            self, "accepted", None if self.accepted is None else bool(self.accepted)
        )
        object.__setattr__(self, "metrics", _scalar_metrics(self.metrics))

    def to_fs(self) -> Dict[str, Any]:
        """Serialize one compact optimization record."""

        payload: Dict[str, Any] = {
            "index": self.index,
            "kind": self.kind,
            "evaluation": self.evaluation,
            "iteration": self.iteration,
            "loss": self.loss.to_fs(),
            "model_digest": self.model_digest,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": dict(self.metrics),
        }
        for key in ("gradient_norm", "step_norm", "step_length", "accepted"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "OptimizationRecord":
        """Deserialize one optimization record."""

        return cls(
            index=data["index"],
            kind=data["kind"],
            evaluation=data["evaluation"],
            iteration=data.get("iteration"),
            loss=LossTerms.from_fs(data["loss"]),
            model_digest=data["model_digest"],
            elapsed_seconds=data["elapsed_seconds"],
            gradient_norm=data.get("gradient_norm"),
            step_norm=data.get("step_norm"),
            step_length=data.get("step_length"),
            accepted=data.get("accepted"),
            metrics=data.get("metrics", {}),
        )


class OptimizationHistory:
    """Append-only, atomically persisted objective and iteration history."""

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
        started_at: Optional[str] = None,
        status: str = "running",
        message: Optional[str] = None,
        records: Optional[Iterable[OptimizationRecord]] = None,
    ):
        if status not in {"running", *_FINAL_STATUSES}:
            raise ValueError("unsupported optimization history status")
        self.path = None if path is None else Path(path)
        self.metadata = _scalar_metrics(metadata)
        self.started_at = started_at or _utc_now()
        self.updated_at = self.started_at
        self.status = status
        self.message = None if message is None else str(message)
        self.records = list(records or ())
        self._started_monotonic = time.monotonic()
        self._validate_records()
        self._elapsed_offset = max(
            (record.elapsed_seconds for record in self.records), default=0.0
        )

    @property
    def evaluation_count(self) -> int:
        """Return the number of recorded objective evaluations."""

        return sum(record.kind == "evaluation" for record in self.records)

    @property
    def iteration_count(self) -> int:
        """Return the number of recorded accepted/current iterates."""

        return sum(record.kind == "iteration" for record in self.records)

    @property
    def evaluations(self) -> tuple[OptimizationRecord, ...]:
        """Return objective-evaluation records in append order."""

        return tuple(record for record in self.records if record.kind == "evaluation")

    @property
    def iterations(self) -> tuple[OptimizationRecord, ...]:
        """Return iteration records in append order."""

        return tuple(record for record in self.records if record.kind == "iteration")

    def losses(self, kind: str = "iteration") -> np.ndarray:
        """Return the total loss sequence for evaluations or iterations."""

        if kind not in {"evaluation", "iteration"}:
            raise ValueError("loss history kind must be evaluation or iteration")
        return np.asarray(
            [record.loss.total for record in self.records if record.kind == kind],
            dtype=np.float64,
        )

    def record_evaluation(
        self,
        model: Union[Sequence[float], np.ndarray],
        loss: LossTerms,
        *,
        gradient_norm: Optional[float] = None,
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> OptimizationRecord:
        """Append one actual objective evaluation and persist it immediately."""

        vector = _real_model(model)
        record = OptimizationRecord(
            index=len(self.records),
            kind="evaluation",
            evaluation=self.evaluation_count,
            iteration=None,
            loss=loss,
            model_digest=_model_digest(vector),
            elapsed_seconds=self._elapsed_seconds(),
            gradient_norm=gradient_norm,
            metrics=metrics or {},
        )
        return self._append(record)

    def record_iteration(
        self,
        model: Union[Sequence[float], np.ndarray],
        loss: LossTerms,
        *,
        gradient_norm: Optional[float] = None,
        step_norm: Optional[float] = None,
        step_length: Optional[float] = None,
        accepted: bool = True,
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> OptimizationRecord:
        """Append one optimizer iterate linked to the current evaluation count."""

        vector = _real_model(model)
        digest = _model_digest(vector)
        previous = self.iterations
        if previous and previous[-1].model_digest == digest:
            return previous[-1]
        record = OptimizationRecord(
            index=len(self.records),
            kind="iteration",
            evaluation=self.evaluation_count,
            iteration=self.iteration_count,
            loss=loss,
            model_digest=digest,
            elapsed_seconds=self._elapsed_seconds(),
            gradient_norm=gradient_norm,
            step_norm=step_norm,
            step_length=step_length,
            accepted=bool(accepted),
            metrics=metrics or {},
        )
        return self._append(record)

    def finish(self, status: str, message: Optional[str] = None) -> None:
        """Mark a run terminal and persist the final status."""

        if status not in _FINAL_STATUSES:
            raise ValueError(
                f"final optimization status must be one of {_FINAL_STATUSES}"
            )
        self.status = status
        self.message = None if message is None else str(message)
        self.updated_at = _utc_now()
        if self.path is not None:
            self.save()

    def resume(self, message: Optional[str] = None) -> None:
        """Reopen a nonconverged history for an explicit checkpoint restart."""

        if self.status == "converged":
            raise RuntimeError("cannot resume a converged optimization history")
        if self.status == "running":
            return
        self.status = "running"
        self.message = None if message is None else str(message)
        self._elapsed_offset = max(
            (record.elapsed_seconds for record in self.records), default=0.0
        )
        self._started_monotonic = time.monotonic()
        self.updated_at = _utc_now()
        if self.path is not None:
            self.save()

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the complete compact history."""

        return {
            "schema": _HISTORY_SCHEMA,
            "status": self.status,
            "message": self.message,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
            "evaluation_count": self.evaluation_count,
            "iteration_count": self.iteration_count,
            "records": [record.to_fs() for record in self.records],
        }

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Atomically save the history and remember its path."""

        if path is not None:
            self.path = Path(path)
        if self.path is None:
            raise ValueError("optimization history has no output path")
        self.updated_at = _utc_now()
        return _write_json_atomic(self.path, self.to_fs())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OptimizationHistory":
        """Load and validate a persisted optimization history."""

        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != _HISTORY_SCHEMA:
            raise ValueError("unsupported optimization history schema")
        history = cls(
            path=path,
            metadata=data.get("metadata"),
            started_at=data.get("started_at"),
            status=data.get("status", "running"),
            message=data.get("message"),
            records=[OptimizationRecord.from_fs(value) for value in data["records"]],
        )
        history.updated_at = str(data.get("updated_at", history.started_at))
        if history.evaluation_count != int(data.get("evaluation_count", -1)):
            raise ValueError("optimization history evaluation count is inconsistent")
        if history.iteration_count != int(data.get("iteration_count", -1)):
            raise ValueError("optimization history iteration count is inconsistent")
        return history

    def _append(self, record: OptimizationRecord) -> OptimizationRecord:
        """Validate and persist one append-only record."""

        if self.status != "running":
            raise RuntimeError("cannot append to a finished optimization history")
        if record.index != len(self.records):
            raise ValueError("optimization record index is not append-only")
        self.records.append(record)
        self.updated_at = _utc_now()
        if self.path is not None:
            self.save()
        return record

    def _elapsed_seconds(self) -> float:
        """Return monotonic elapsed time, including time recorded before reload."""

        return self._elapsed_offset + time.monotonic() - self._started_monotonic

    def _validate_records(self) -> None:
        """Verify counters and append order after construction or loading."""

        evaluation = 0
        iteration = 0
        for index, record in enumerate(self.records):
            if record.index != index:
                raise ValueError("optimization history record indices are inconsistent")
            if record.kind == "evaluation":
                if record.evaluation != evaluation or record.iteration is not None:
                    raise ValueError(
                        "optimization evaluation counters are inconsistent"
                    )
                evaluation += 1
            else:
                if record.evaluation != evaluation or record.iteration != iteration:
                    raise ValueError("optimization iteration counters are inconsistent")
                iteration += 1


@dataclass(frozen=True)
class OptimizationCheckpoint:
    """Minimal real-model checkpoint needed to restart an optimizer."""

    model: np.ndarray
    iteration: int
    evaluations: int
    loss: LossTerms
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _real_model(self.model))
        if int(self.iteration) < 0 or int(self.evaluations) < 0:
            raise ValueError("checkpoint counters must be nonnegative")
        object.__setattr__(self, "iteration", int(self.iteration))
        object.__setattr__(self, "evaluations", int(self.evaluations))
        object.__setattr__(self, "metadata", _scalar_metrics(self.metadata))

    def save(self, path: Union[str, Path]) -> Path:
        """Atomically write the restart checkpoint as portable float64 HDF5."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp.h5", dir=path.parent
        )
        os.close(descriptor)
        try:
            with h5py.File(temporary, "w") as h5:
                h5.attrs["schema"] = _CHECKPOINT_SCHEMA
                h5.attrs["iteration"] = self.iteration
                h5.attrs["evaluations"] = self.evaluations
                h5.attrs["metadata"] = json.dumps(dict(self.metadata), sort_keys=True)
                h5.create_dataset("model", data=self.model, dtype=np.float64)
                loss = h5.create_group("loss")
                for key, value in self.loss.to_fs().items():
                    loss.attrs[key] = value
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OptimizationCheckpoint":
        """Read and validate a restart checkpoint."""

        with h5py.File(path, "r") as h5:
            if h5.attrs.get("schema") != _CHECKPOINT_SCHEMA:
                raise ValueError("unsupported optimization checkpoint schema")
            loss = LossTerms(
                data=float(h5["loss"].attrs["data"]),
                regularization=float(h5["loss"].attrs["regularization"]),
            )
            if not np.isclose(float(h5["loss"].attrs["total"]), loss.total):
                raise ValueError("checkpoint loss total is inconsistent")
            return cls(
                model=np.asarray(h5["model"]),
                iteration=int(h5.attrs["iteration"]),
                evaluations=int(h5.attrs["evaluations"]),
                loss=loss,
                metadata=json.loads(h5.attrs.get("metadata", "{}")),
            )


@dataclass(frozen=True)
class OptimizationResult:
    """Portable terminal summary linked to history and checkpoint artifacts."""

    success: bool
    status: int
    message: str
    model: np.ndarray
    loss: LossTerms
    iterations: int
    evaluations: int
    history: Optional[str] = None
    checkpoint: Optional[str] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _real_model(self.model))
        if int(self.iterations) < 0 or int(self.evaluations) < 0:
            raise ValueError("optimization result counters must be nonnegative")
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "evaluations", int(self.evaluations))
        object.__setattr__(self, "status", int(self.status))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "metrics", _scalar_metrics(self.metrics))

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the terminal summary, including the final small model vector."""

        return {
            "schema": _RESULT_SCHEMA,
            "success": bool(self.success),
            "status": self.status,
            "message": self.message,
            "model": self.model.tolist(),
            "loss": self.loss.to_fs(),
            "iterations": self.iterations,
            "evaluations": self.evaluations,
            "history": self.history,
            "checkpoint": self.checkpoint,
            "metrics": dict(self.metrics),
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Atomically write the terminal JSON summary."""

        return _write_json_atomic(Path(path), self.to_fs())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "OptimizationResult":
        """Load a terminal optimization summary."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema") != _RESULT_SCHEMA:
            raise ValueError("unsupported optimization result schema")
        return cls(
            success=data["success"],
            status=data["status"],
            message=data["message"],
            model=data["model"],
            loss=LossTerms.from_fs(data["loss"]),
            iterations=data["iterations"],
            evaluations=data["evaluations"],
            history=data.get("history"),
            checkpoint=data.get("checkpoint"),
            metrics=data.get("metrics", {}),
        )
