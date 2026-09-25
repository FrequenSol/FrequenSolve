"""Prepare physical-source spectra for Sauce's existing acquisition contract."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
from numpy.typing import ArrayLike

from frequensolve.seismic.spectra import TransferFunction, _axis
from frequensolve.util.mixins import ExportContext

__all__ = ["SourceSignature", "ReceiverTransferFunction", "ReceiverResponse"]


class _PointSpectrum:
    """A shared signal or an explicit one-based physical-source signal map.

    Supply the solve frequencies explicitly; export never uses a plotting grid.
    ``laplace`` lists nonpositive imaginary frequencies in Hz (default zero).
    Every physical source needs a signal; use GainDelay() for an unchanged one.
    Signals must be dimensionless multipliers of the authored source strength.
    ``spectral_derivative='total'`` exports the exact evaluator's dq/df;
    ``'frozen'`` explicitly requests zero derivative in Sauce.
    """

    point_axis = "source"

    def __init__(
        self,
        signals: TransferFunction | Mapping[int, TransferFunction],
        *,
        frequencies: ArrayLike,
        laplace: ArrayLike = (0.0,),
        spectral_derivative: str = "total",
        batch_bytes: int = 64 * 1024**2,
    ) -> None:
        frequencies = _axis(frequencies, "frequencies", nonnegative=True)
        damping = _axis(laplace, "laplace")
        if (
            np.any(np.diff(frequencies) <= 0)
            or len(np.unique(damping)) != len(damping)
            or np.any(damping > 0)
        ):
            raise ValueError(
                "Frequencies must increase; Laplace coordinates must be unique and nonpositive"
            )
        # Sauce matches coordinates with a relative tolerance. Reject axes whose
        # authored coordinates already match more than one row on that backend.
        for axis in (frequencies, np.sort(damping)):
            gap = np.diff(axis)
            tolerance = 1e-6 * np.maximum(np.abs(axis), 1e-8)
            if np.any(gap <= np.maximum(tolerance[:-1], tolerance[1:])):
                raise ValueError(
                    "Spectrum coordinates are ambiguous within Sauce's matching tolerance"
                )
        if spectral_derivative not in {"total", "frozen"}:
            raise ValueError("spectral_derivative must be 'total' or 'frozen'")
        if not isinstance(batch_bytes, int) or batch_bytes <= 0:
            raise ValueError("batch_bytes must be a positive integer")
        values: Iterable[TransferFunction]
        if isinstance(signals, Mapping):
            signals = dict(signals)
            if not signals or any(
                not isinstance(i, (int, np.integer)) or isinstance(i, bool) or i < 1
                for i in signals
            ):
                raise ValueError(
                    "Signature keys must be positive integer physical-source IDs"
                )
            values = signals.values()
        else:
            values = [signals]
        if any(not isinstance(s, TransferFunction) or s.units != "1" for s in values):
            raise ValueError(
                "Source signatures require dimensionless TransferFunction objects; normalize physical samples with relative_to"
            )
        self.signals: TransferFunction | Mapping[int, TransferFunction] = (
            MappingProxyType(signals) if isinstance(signals, dict) else signals
        )
        self.frequencies, self.laplace = frequencies, damping
        self.spectral_derivative = spectral_derivative
        self.batch_bytes = batch_bytes

    def __deepcopy__(self, memo: dict[int, Any]) -> _PointSpectrum:
        # Built-in responses and coordinate arrays are immutable.
        import copy

        signals = (
            dict(self.signals) if isinstance(self.signals, Mapping) else self.signals
        )
        return type(self)(
            copy.deepcopy(signals, memo),
            frequencies=self.frequencies,
            laplace=self.laplace,
            spectral_derivative=self.spectral_derivative,
            batch_bytes=self.batch_bytes,
        )

    def _signals(self, source_count: int) -> list[TransferFunction]:
        if (
            not isinstance(source_count, (int, np.integer))
            or not 0 < source_count < 2**31
        ):
            raise ValueError(
                "Source signatures require a known positive int32 source count"
            )
        if isinstance(self.signals, Mapping):
            if set(self.signals) != set(range(1, source_count + 1)):
                raise ValueError(
                    "Signature IDs must exactly match physical sources 1..source_count"
                )
            return [self.signals[i] for i in range(1, source_count + 1)]
        return [self.signals] * source_count

    def _to_fs(self, ctx: ExportContext, *, source_count: int) -> dict[str, Any]:
        """Write an immutable content-addressed artifact and return its reference.

        Uses bounded frequency blocks, with one full physical-source row as the
        minimum. Publication occurs only after samples and derivatives validate.
        Exported artifacts are independent of the mutable simulation array store.
        """
        signals = self._signals(source_count)
        directory = ctx.store.path.parent if ctx.store is not None else ctx.path
        if directory is None:
            raise ValueError("SourceSignature export requires an artifact directory")
        directory = Path(directory) / f"{self.point_axis}-spectra"
        directory.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=".signature-", suffix=".h5", dir=directory
        )
        os.close(handle)
        temporary = Path(temporary_name)
        digest = hashlib.sha256()
        has_damping = len(self.laplace) != 1 or self.laplace[0] != 0
        dims = (len(self.frequencies), source_count, 2)
        shape = (len(self.laplace), *dims) if has_damping else dims
        names = ["q"] + (["q_f"] if self.spectral_derivative == "total" else [])
        data_hashes = {name: hashlib.sha256() for name in names}
        metadata = dict(
            schema="fs-acquisition-spectrum-1",
            point_axis=self.point_axis,
            spectral_derivative=self.spectral_derivative,
            units="1",
            transform="exp(-2*pi*i*(f+i*laplace)*t)",
            application=f"physical_{self.point_axis}_before_encoding",
        )
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        for axis in (
            self.frequencies,
            self.laplace,
            np.arange(1, source_count + 1, dtype=np.int64),
        ):
            digest.update(np.asarray(axis, dtype="<f8").tobytes())
        # Deduplicate shared definitions without comparing array-valued objects.
        groups: dict[int, tuple[TransferFunction, list[int]]] = {}
        for column, signal in enumerate(signals):
            groups.setdefault(id(signal), (signal, []))[1].append(column)
        definitions = [
            json.dumps(signal.describe(), sort_keys=True)
            for signal, columns in groups.values()
        ]
        definition_ids = np.empty(source_count, dtype="<i8")
        for index, ((signal, columns), definition) in enumerate(
            zip(groups.values(), definitions), 1
        ):
            definition_ids[columns] = index
            digest.update(definition.encode() + b"\n")
        digest.update(definition_ids.tobytes())
        width = max(1, min(256, self.batch_bytes // max(1, source_count * 32)))
        try:
            with h5py.File(temporary, "w") as h5:
                h5.attrs.update(metadata)
                # Banks can exceed HDF5's compact attribute size limit.
                h5.create_dataset(
                    "signal_definitions", data=definitions, dtype=h5py.string_dtype()
                )
                h5["signal_definition_ids"] = definition_ids
                h5["frequency"] = self.frequencies
                h5[f"{self.point_axis}_ids"] = np.arange(
                    1, source_count + 1, dtype=np.int64
                )
                if has_damping:
                    h5["laplace"] = self.laplace
                chunks: tuple[int, ...] = (1, min(source_count, 8192), 2)
                if has_damping:
                    chunks = (1, *chunks)
                datasets = {
                    name: h5.create_dataset(
                        name, shape=shape, dtype="<f4", chunks=chunks
                    )
                    for name in names
                }
                for damping_index, damping in enumerate(self.laplace):
                    for first in range(0, len(self.frequencies), width):
                        last = min(first + width, len(self.frequencies))
                        for name in names:
                            values = np.empty(
                                (last - first, source_count), dtype=np.complex64
                            )
                            for signal, columns in groups.values():
                                samples = signal.at_frequencies(
                                    self.frequencies[first:last],
                                    laplace=damping,
                                    derivative=name == "q_f",
                                )
                                if np.shape(samples) != (last - first,):
                                    raise ValueError(
                                        "Response evaluator returned the wrong shape"
                                    )
                                with np.errstate(over="ignore", invalid="ignore"):
                                    values[:, columns] = np.asarray(samples)[:, None]
                            if not np.all(np.isfinite(values)):
                                raise ValueError(
                                    "Source spectrum exceeds finite float32 storage"
                                )
                            packed = values.view(np.float32).reshape(
                                last - first, source_count, 2
                            )
                            data_hashes[name].update(
                                np.asarray(packed, dtype="<f4").tobytes()
                            )
                            if has_damping:
                                datasets[name][damping_index, first:last] = packed
                            else:
                                datasets[name][first:last] = packed
                for name in names:
                    digest.update(name.encode())
                    digest.update(data_hashes[name].digest())
                h5.attrs["content_hash"] = digest.hexdigest()
            path = directory / f"{digest.hexdigest()}.h5"
            # Atomic replacement is safe: the destination identifies identical content.
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        result = dict(
            file=str(ctx.relative_to_project(path)),
            dataset="q",
            frequencies_dataset="frequency",
            spectral_derivative=self.spectral_derivative,
            hash="sha256:" + digest.hexdigest(),
        )
        result[f"{self.point_axis}_ids_dataset"] = f"{self.point_axis}_ids"
        if has_damping:
            result["laplace_damping_dataset"] = "laplace"
        if self.spectral_derivative == "total":
            result["frequency_derivative_dataset"] = "q_f"
        return result


class SourceSignature(_PointSpectrum):
    """A dimensionless shared signal or one-based physical-source signal map.

    Supply explicit solve frequencies and nonpositive Laplace coordinates.
    ``spectral_derivative='total'`` exports dq/df; ``'frozen'`` exports zero.
    Every physical source must have a signal; GainDelay() is the identity.
    """

    def to_fs(self, ctx: ExportContext, *, source_count: int) -> dict[str, Any]:
        """Export the immutable physical-source spectrum table."""
        return self._to_fs(ctx, source_count=source_count)


class ReceiverTransferFunction(_PointSpectrum):
    """Diagonal transfer function H(f) at physical receiver nodes.

    Maps ideal sampled fields to measured fields by Y(f) = H(f) X(f), before
    receiver encoding. ``ReceiverResponse`` is a compatibility alias.

    Uses the same spectral evaluators as SourceSignature. A shared response
    applies to every node; mapping keys are one-based physical-node IDs in the
    group's expanded coordinate order, before array reduction or encoding.
    Responses preserve the component's physical output units (dimensionless
    relative calibration). Cross-component matrices are not represented here.
    """

    point_axis = "receiver"

    def to_fs(self, ctx: ExportContext, *, receiver_count: int) -> dict[str, Any]:
        return self._to_fs(ctx, source_count=receiver_count)


ReceiverResponse = ReceiverTransferFunction
