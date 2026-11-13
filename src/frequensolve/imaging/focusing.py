"""Frequency-coherent time-reversal focusing objectives.

Back-propagating the observed data ``o`` of source ``s`` and sampling it at the
source is, by reciprocity, ``C_f,s = sum_r conj(d_f,s(r)) o_f,s(r)`` with ``d``
the modeled data of that source.  Summing frequencies coherently gives the
refocused trace ``A_s(tau) = sum_f C_f,s exp(2 pi i f tau)``; its energy in a
Gaussian lag window of standard deviation ``sigma`` is exactly

``E_s = Re(C_s^H K C_s)``, ``K_ff' = exp(-2 pi^2 sigma^2 (f - f')^2)``

(real parts of the frequencies; ``sigma = 0`` is the zero-lag refocus).  By
Cauchy-Schwarz the focusing ratio ``rho_s = E_s / (|d_s|^2 |o_s|^2)`` lies in
``[0, 1]`` and equals one at zero lag when the modeled data are proportional
to the observed data; the objective is the mean defocus

``J = 1 - mean_s rho_s``,

nonnegative (as the FWI workflow's loss bookkeeping requires) and smallest
when the modeled data are proportional to the observed data.  Gradients use the Jacobian of the modeled
data only (``Re(J_d^H G)``, ``G = 2 dJ/d conj(d)``).

:class:`SourceAperture` softens the focus spatially around every source with a
cosine-tapered node grid, two ways:

- ``strategy="linear"`` (average, then square): each source becomes one
  extended source ``D_s = sum_k w_k d_s,k`` (one encoded right-hand side), and
  the objective above is evaluated on ``D``.
- ``strategy="pointwise"`` (WEFT: square, then integrate):
  ``J = 1 - mean_s sum_k wbar_k E_s,k / (|d_s,k|^2 |o_s|^2)``. Matching
  correlations and energies are evaluated at a coarse node subset.
  The normalized ratios are interpolated to the tapered grid; the gradient
  uses the transpose of this same interpolation.

Both softenings bias the kinematics toward models in which an off-center source
mimics the real one; the point focus (no aperture) is the unbiased objective.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

import h5py
import numpy as np

from .controls import (
    _MATERIAL_KINDS,
    ControlSpace,
    ControlState,
    ControlVector,
)
from .data import ObservedData
from .misfit import Misfit, Normalization, ObjectiveTerm

__all__ = ["Focusing", "FocusingLinearization", "FocusingProblem", "SourceAperture"]

_STRATEGIES = ("linear", "pointwise")


def _magnitude(value: Any, units: str, name: str) -> float:
    """Return ``value`` in ``units`` (plain numbers are taken as ``units``)."""

    if hasattr(value, "to") and hasattr(value, "magnitude"):
        number = float(value.to(units).magnitude)
    else:
        number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def lag_kernel(frequencies: Sequence[Any], window: float) -> np.ndarray:
    """Return ``K_ff' = exp(-2 pi^2 sigma^2 (f - f')^2)`` on the real frequency parts."""

    f = np.array([complex(value).real for value in frequencies], dtype=np.float64)
    return np.exp(-2.0 * np.pi**2 * window**2 * (f[:, None] - f[None, :]) ** 2)


def coherent_focus(
    d: np.ndarray,
    o: np.ndarray,
    rows: Sequence[Tuple[np.ndarray, np.ndarray]],
    kernel: np.ndarray,
    source_weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return ``(J, G, ratio)`` of the coherent focus over modeled/observed rows.

    Args:
        d, o: Modeled and observed values in one row ordering.
        rows: Per frequency ``(indices, source)`` with zero-based source ids.
        kernel: Lag kernel over the same frequencies.

    ``G = 2 dJ/d conj(d)``; sources with no modeled or observed energy do not
    enter the mean.
    """

    n_f = len(rows)
    n_s = 1 + max(int(source.max()) for _, source in rows if source.size)
    C = np.zeros((n_f, n_s), complex)
    Nd = np.zeros(n_s)
    No = np.zeros(n_s)
    for t, (index, source) in enumerate(rows):
        np.add.at(C[t], source, np.conj(d[index]) * o[index])
        np.add.at(Nd, source, np.abs(d[index]) ** 2)
        np.add.at(No, source, np.abs(o[index]) ** 2)
    valid = (Nd > 0) & (No > 0)
    if not np.any(valid):
        raise ValueError("focusing needs nonzero modeled and observed data")
    KC = kernel @ C
    E = np.real(np.sum(np.conj(C) * KC, axis=0))
    c = np.where(valid, Nd * No, 1.0)
    ratio = np.where(valid, E / c, 0.0)
    weights = (
        np.ones(n_s) if source_weights is None else np.asarray(source_weights, float)
    )
    if (
        weights.shape != (n_s,)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
    ):
        raise ValueError(
            "source weights must be finite, nonnegative and match the sources"
        )
    weights = np.where(valid, weights, 0.0)
    if weights.sum() <= 0:
        raise ValueError("focusing needs positive weight on nonzero data")
    weights = weights / weights.sum()
    scale = -2.0 * weights
    G = np.zeros_like(d)
    for t, (index, source) in enumerate(rows):
        ok = valid[source]
        s = source[ok]
        idx = index[ok]
        G[idx] = scale[s] * (
            np.conj(KC[t, s]) * o[idx] / c[s] - ratio[s] * d[idx] / Nd[s]
        )
    # Clamp roundoff at the mathematical bounds, after using the exact derivative.
    ratio = np.clip(ratio, 0.0, 1.0)
    return max(0.0, 1.0 - float(weights @ ratio)), G, ratio


def _rows(lin: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return per frequency ``(indices, zero-based encoded source)`` of every objective row."""

    out = []
    for frequency in lin.frequencies:
        layouts = lin.data_space.term_layouts(frequency=frequency)
        index = np.concatenate([layout.indices for layout in layouts])
        source = (
            np.concatenate([layout.coordinate_keys[:, 0] for layout in layouts]) - 1
        )
        out.append((np.asarray(index, int), np.asarray(source, int)))
    return out


def unit_misfit(groups: Sequence[str]) -> Misfit:
    """Unit-weight waveform L2 (explicit unit scale, ``sum``): its rows are raw data."""

    normalization = Normalization(kind="explicit", value=1.0, reduction="sum")
    return Misfit.terms(
        *[
            ObjectiveTerm(
                group, loss="l2", comparison="waveform", normalization=normalization
            )
            for group in groups
        ]
    )


# ---------------------------------------------------------------------------
# specifications
# ---------------------------------------------------------------------------


class SourceAperture:
    """Cosine-tapered node grid around every source for spatial softening.

    Args:
        half_width: Taper half-width ``h`` (length; plain numbers are in the
            source geometry's units).  Node weights are
            ``prod_i (1 + cos(pi x_i / h)) / 2``.
        spacing: Node spacing (``h / spacing`` nodes per half axis).
        coarse: Nodes per axis of the coarse subset on which the pointwise
            strategy evaluates normalized focus (``>= 2``; ratios are
            interpolated multilinearly elsewhere).
        bounds: Optional per-axis ``(minimum, maximum)`` absolute coordinates
            (geometry units, ``None`` for open); nodes outside are dropped,
            e.g. ``[(None, None), (0.001, None)]`` keeps nodes below a free
            surface at depth zero.
    """

    def __init__(
        self,
        half_width: Any,
        spacing: Any,
        *,
        coarse: int = 3,
        bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    ) -> None:
        self.half_width = half_width
        self.spacing = spacing
        if int(coarse) < 2:
            raise ValueError("coarse needs at least two nodes per axis")
        self.coarse = int(coarse)
        self.bounds = None if bounds is None else [tuple(b) for b in bounds]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "half_width": str(self.half_width),
            "spacing": str(self.spacing),
            "coarse": self.coarse,
            "bounds": self.bounds,
        }

    def offsets(self, dim: int, units: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return node offsets ``(K, dim)`` and taper weights ``(K,)`` in ``units``."""

        h = _magnitude(self.half_width, units, "half_width")
        step = _magnitude(self.spacing, units, "spacing")
        if h <= 0 or step <= 0 or step > h:
            raise ValueError("aperture needs 0 < spacing <= half_width")
        n = int(round(h / step))
        axis = np.arange(-(n - 1), n) * (h / n)
        grids = np.meshgrid(*([axis] * dim), indexing="ij")
        offsets = np.stack([g.ravel() for g in grids], 1)
        weights = np.prod(0.5 * (1.0 + np.cos(np.pi * offsets / h)), axis=1)
        return offsets, weights

    def nodes(
        self, centers: np.ndarray, units: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return kept offsets, weights and the coarse subset for ``centers`` (all sources)."""

        offsets, weights = self.offsets(centers.shape[1], units)
        keep = weights > 0
        if self.bounds is not None:
            if len(self.bounds) != centers.shape[1]:
                raise ValueError("aperture bounds need one (min, max) per axis")
            for axis, (low, high) in enumerate(self.bounds):
                position = centers[:, axis][:, None] + offsets[:, axis][None, :]
                if low is not None:
                    keep &= np.all(position >= float(low), axis=0)
                if high is not None:
                    keep &= np.all(position <= float(high), axis=0)
        if not np.any(keep):
            raise ValueError("the aperture keeps no nodes inside its bounds")
        offsets, weights = offsets[keep], weights[keep]
        axes = []
        for axis in range(offsets.shape[1]):
            values = np.unique(np.round(offsets[:, axis], 12))
            pick = np.linspace(0, len(values) - 1, min(self.coarse, len(values)))
            axes.append(values[np.unique(np.round(pick).astype(int))])
        coarse = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], 1)
        return offsets, weights, coarse


class Focusing:
    """Specification of a frequency-coherent focusing objective.

    Args:
        window: Standard deviation of the Gaussian lag window (seconds, or a
            time quantity); ``0`` measures the zero-lag refocus only.
        aperture: Optional :class:`SourceAperture` for spatial softening.
        strategy: ``"linear"`` (average, then square) or ``"pointwise"``
            (square, then integrate); only used with an aperture.
    """

    def __init__(
        self,
        window: Any = 0.1,
        *,
        aperture: Optional[SourceAperture] = None,
        strategy: str = "linear",
    ) -> None:
        self.window = _magnitude(window, "s", "window")
        if self.window < 0:
            raise ValueError("window must be nonnegative")
        if aperture is not None and not isinstance(aperture, SourceAperture):
            raise TypeError("aperture must be a SourceAperture")
        self.aperture = aperture
        strategy = str(strategy).strip().lower()
        if strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {_STRATEGIES}")
        self.strategy = strategy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window,
            "strategy": self.strategy,
            "aperture": None if self.aperture is None else self.aperture.to_dict(),
        }

    def __repr__(self) -> str:
        return f"Focusing(window={self.window:g} s, aperture={self.aperture is not None}, strategy={self.strategy!r})"


# ---------------------------------------------------------------------------
# acquisition helpers
# ---------------------------------------------------------------------------


def _source_points(simulation: Any) -> Tuple[np.ndarray, str, Optional[str], str]:
    """Return physical source coordinates, their units, system and kind."""

    from frequensolve.seismic.sources import _coordinate_array_with_metadata
    from frequensolve.units import unit_expression, ureg

    geometry = simulation.acquisition.source_geometry
    if geometry is None:
        raise ValueError("focusing needs a source geometry")
    kind = str(getattr(geometry, "kind", "scalar"))
    if kind not in {"scalar", "monopole"}:
        raise NotImplementedError("spatial focusing supports scalar point sources only")
    default_units = simulation.units.defaults.get("length", "km")
    units = unit_expression(geometry.units or default_units)
    system = geometry.system
    coordinates: list[np.ndarray] | np.ndarray
    if geometry.geometry_type == "Inline":
        rows = [
            _coordinate_array_with_metadata(value)
            for value in geometry.coordinate_values()
        ]
        units = unit_expression(rows[0][1] or default_units)
        system = rows[0][2]
        coordinates = []
        for values, row_units, row_system in rows:
            if (row_system or "global") != (system or "global"):
                raise ValueError("aperture sources must use one coordinate system")
            coordinates.append(
                ureg.Quantity(values, unit_expression(row_units or default_units))
                .to(units)
                .magnitude
            )
    elif geometry.geometry_type == "HDF5":
        with h5py.File(geometry.file, "r") as h5:
            coordinates = np.asarray(h5[geometry.dataset], dtype=np.float64)
    else:
        raise NotImplementedError(
            f"{geometry.geometry_type} source geometry is not supported"
        )
    return np.asarray(coordinates, dtype=np.float64), units, system, kind


def _source_rows(simulation: Any, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return per encoded source its physical source index and real weight.

    Spatial focusing needs every right-hand side to fire one physical source
    with a real weight (identity, selection or scaled selections).
    """

    encoding = simulation.acquisition.source_encoding
    if encoding is None:
        return np.arange(n_points), np.ones(n_points)
    if encoding.frequencies is not None:
        raise NotImplementedError(
            "spatial focusing needs a frequency-independent encoding"
        )
    weights = encoding.weights
    if weights is None and encoding.encoding_type == "HDF5Dense":
        with h5py.File(encoding.file, "r") as h5:
            raw = np.asarray(h5[encoding.dataset], dtype=np.float64)
        weights = raw[..., 0] + 1j * raw[..., 1]
    if weights is None:
        raise NotImplementedError(
            f"{encoding.encoding_type} source encodings are not supported"
        )
    weights = np.asarray(weights)
    nonzero = weights != 0
    if np.any(nonzero.sum(1) != 1):
        raise NotImplementedError(
            "spatial focusing needs one physical source per right-hand side"
        )
    index = np.argmax(nonzero, axis=1)
    value = weights[np.arange(len(index)), index]
    if np.any(np.abs(np.imag(value)) > 1e-12 * np.abs(value)):
        raise NotImplementedError("spatial focusing needs real source-encoding weights")
    return index, np.real(value)


def _swap_sources(
    simulation: Any,
    name: str,
    points: np.ndarray,
    units: str,
    system: Optional[str],
    kind: str,
    encoding: Any,
    *,
    source_indices: np.ndarray,
    directory: Path,
) -> Any:
    from frequensolve.seismic.sources import SourceGeometry

    copy = simulation.copy(name)
    copy.acquisition.set_sources(
        SourceGeometry.points(
            kind=kind,
            coords=points,
            units=units,
            system=system,
            names=[f"focus_{i}" for i in range(len(points))],
        )
    )
    copy.acquisition.set_source_encoding(encoding)
    if copy.acquisition.source_signature is not None:
        copy.acquisition.source_signature = _node_signature(
            copy.acquisition.source_signature,
            Path(copy.project_path),
            source_indices,
            directory,
        )
    return copy


def _node_signature(
    signature: Mapping[str, Any],
    project: Path,
    source_indices: np.ndarray,
    directory: Path,
) -> Dict[str, Any]:
    """Repeat/reorder physical spectra in bounded blocks for aperture nodes."""

    result = dict(signature)
    source_file = Path(result["file"])
    if not source_file.is_absolute():
        source_file = project / source_file
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "source-signature.h5"
    temporary = destination.with_suffix(".tmp.h5")
    digest = hashlib.sha256()
    try:
        with (
            h5py.File(source_file, "r") as original,
            h5py.File(temporary, "w") as output,
        ):
            output.attrs.update(original.attrs)
            for key in ("frequencies_dataset", "laplace_damping_dataset"):
                if key in result:
                    data = original[result[key]][:]
                    output.create_dataset(result[key], data=data)
                    digest.update(data.tobytes())
            ids = np.arange(1, len(source_indices) + 1, dtype=np.int64)
            output.create_dataset(result["source_ids_dataset"], data=ids)
            digest.update(ids.tobytes())
            for key in ("dataset", "frequency_derivative_dataset"):
                if key not in result:
                    continue
                dataset = original[result[key]]
                target = output.create_dataset(
                    result[key],
                    shape=(*dataset.shape[:-2], len(source_indices), 2),
                    dtype=dataset.dtype,
                    chunks=True,
                )
                digest.update(key.encode())
                for prefix in np.ndindex(dataset.shape[:-2]):
                    for first in range(0, len(source_indices), 256):
                        last = min(first + 256, len(source_indices))
                        unique, inverse = np.unique(
                            source_indices[first:last], return_inverse=True
                        )
                        values = dataset[(*prefix, unique, slice(None))][inverse]
                        target[(*prefix, slice(first, last), slice(None))] = values
                        digest.update(values.tobytes())
            output.attrs["content_hash"] = digest.hexdigest()
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    result.update(file=str(destination), hash="sha256:" + digest.hexdigest())
    return result


# ---------------------------------------------------------------------------
# linearizations and the problem view
# ---------------------------------------------------------------------------


class FocusingLinearization:
    """Focusing objective value and gradient at one point.

    Attributes:
        value: Objective value (mean defocus ``1 - ratio``).
        gradient: Derivative on the problem's active space. Aperture views
            require material controls; point focus supports all active controls.
        ratio: Focusing ratio per encoded source (per node for ``pointwise``).
        point, state, space: The linearization point.
    """

    def __init__(
        self,
        problem: "FocusingProblem",
        *,
        state: ControlState,
        value: float,
        gradient: Optional[ControlVector],
        ratio: np.ndarray,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.problem = problem
        self.space = problem.space
        self.state = state
        self.point = state.vector(self.space)
        self.value = float(value)
        self.gradient = gradient
        self.ratio = ratio
        self.details = dict(details or {})

    @property
    def normal(self) -> Any:
        raise NotImplementedError(
            "focusing objectives have no Gauss-Newton normal operator"
        )

    @property
    def jacobian(self) -> Any:
        raise NotImplementedError("focusing objectives have no data-space Jacobian")

    def __repr__(self) -> str:
        return f"FocusingLinearization(value={self.value:.6g}, gradient={self.gradient is not None})"


class _Auxiliary:
    """Grid-source problems shared by every view of one focusing problem."""

    def __init__(self) -> None:
        self.problems: Dict[str, Any] = {}
        self.namespace = uuid4().hex[:16]


class FocusingProblem:
    """An :class:`ImagingProblem` view whose objective is coherent focusing.

    Shares the problem's state, cache, site and simulation and satisfies the
    objective protocol of :class:`~frequensolve.imaging.workflows.FWI`
    (``restrict``, ``linearize``, ``value``, ``gradient``, ``state``,
    ``space``, ``vector``); it has no normal operator, so use a first-order
    optimizer.  The view's misfit is replaced by a unit-weight waveform L2 so
    Sauce's saved rows are the raw modeled and observed data.
    """

    def __init__(
        self, problem: Any, focusing: Focusing, *, _aux: Optional[_Auxiliary] = None
    ) -> None:
        if not isinstance(focusing, Focusing):
            raise TypeError("focusing must be a Focusing")
        groups = [group.name for group in problem.observed_groups]
        self._problem = problem.restrict(
            misfit=unit_misfit(groups), kernel_derivative=None
        )
        self.focusing = focusing
        if focusing.aperture is not None:
            if focusing.strategy == "pointwise" and any(
                group.survey is not None
                for group in problem.simulation.acquisition.receiver_groups
            ):
                raise NotImplementedError(
                    "pointwise aperture focusing requires dense receiver sampling; "
                    "use linear aperture focusing for sparse surveys"
                )
            if any(b.kind not in _MATERIAL_KINDS for b in self.space.resolved_blocks):
                raise ValueError(
                    "aperture focusing requires an active material-only space; restrict(active=...) first"
                )
        self._aux = _Auxiliary() if _aux is None else _aux
        self._cache: OrderedDict[Tuple[str, bool], FocusingLinearization] = (
            OrderedDict()
        )

    # -- delegation -----------------------------------------------------------

    @property
    def problem(self) -> Any:
        return self._problem

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._problem, name)

    @property
    def state(self) -> Optional[ControlState]:
        return self._problem.state

    @state.setter
    def state(self, value: ControlState) -> None:
        self._validate_state(value)
        self._problem.state = value

    @property
    def space(self) -> ControlSpace:
        return self._problem.space

    @property
    def smoothing(self) -> None:
        """Focusing gradients carry no Sauce smoothing; always ``None``."""

        return None

    def identity(self) -> Dict[str, Any]:
        return {**self._problem.identity(), "focusing": self.focusing.to_dict()}

    def __repr__(self) -> str:
        return (
            f"FocusingProblem(name={self._problem.name!r}, {self.focusing!r}, "
            f"blocks={list(self.space.blocks)}, frequencies={self._problem.frequencies})"
        )

    def restrict(
        self,
        *args: Any,
        misfit: Any = None,
        loss: Any = None,
        smoothing: Any = None,
        **kwargs: Any,
    ) -> "FocusingProblem":
        """Return a stage view; the focusing objective replaces any misfit or loss."""

        if misfit is not None or loss is not None:
            raise ValueError(
                "a focusing problem's objective cannot be replaced by a misfit"
            )
        view = self._problem.restrict(*args, **kwargs)
        return FocusingProblem(view, self.focusing, _aux=self._aux)

    def vector(self, state: Optional[ControlState] = None) -> ControlVector:
        return self._problem.vector(state)

    def with_controls(self, controls: Any, **kwargs: Any) -> "FocusingProblem":
        """Transfer the model layout while preserving the focusing objective."""

        # Auxiliary acquisitions depend on the control basis and need rebuilding.
        return FocusingProblem(
            self._problem.with_controls(controls, **kwargs), self.focusing
        )

    def state_from(self, vector: Any) -> ControlState:
        return self._problem.state_from(vector)

    # -- objective ------------------------------------------------------------

    def linearize(
        self, v: Any = None, *, gradient: bool = True
    ) -> FocusingLinearization:
        """Return the focusing value (and gradient) at ``v``."""

        state = self._problem._state_at(v)
        self._validate_state(state)
        key = (
            hashlib.sha256(np.ascontiguousarray(state.values).tobytes()).hexdigest(),
            bool(gradient),
        )
        if not gradient and (key[0], True) in self._cache:
            key = (key[0], True)
        if key not in self._cache:
            if self.focusing.aperture is None:
                self._cache[key] = self._point(state, gradient)
            elif self.focusing.strategy == "linear":
                self._cache[key] = self._linear(state, gradient)
            else:
                self._cache[key] = self._pointwise(state, gradient)
            while len(self._cache) > self._problem._shared.cache.capacity:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def value(self, v: Any = None) -> float:
        return self.linearize(v, gradient=False).value

    def gradient(self, v: Any = None) -> ControlVector:
        lin = self.linearize(v, gradient=True)
        assert lin.gradient is not None
        return lin.gradient

    def _validate_state(self, state: ControlState) -> None:
        """Aperture acquisitions retain the authored non-material controls."""

        if self.focusing.aperture is None:
            return
        self._problem._ensure_registry()
        authored = self._problem._shared.authored
        for block in self._problem.full_space.resolved_blocks:
            if block.kind in _MATERIAL_KINDS:
                continue
            sl = state.space.full_slices[block.name]
            if not np.array_equal(state.values[sl], authored.values[sl]):
                raise ValueError(
                    f"aperture focusing holds {block.name!r} at its authored value"
                )

    def _stage_key(self, tag: str) -> str:
        """Separate acquisition jobs and observed rows across frequency stages."""

        frequencies = np.asarray(self._problem.frequencies, dtype=np.complex128)
        digest = hashlib.sha256(frequencies.tobytes()).hexdigest()[:16]
        return f"{tag}_{self._aux.namespace}_{digest}"

    def _observations(self, lin: Any, repeats: int = 1) -> np.ndarray:
        """Map original encoded observations to auxiliary rows by coordinate identity."""

        key = self._stage_key("observed")
        if key not in self._aux.problems:
            base = self._problem.linearize(gradient=False)
            self._aux.problems[key] = (base.data_space, base.observed().values.copy())
        space, observed = self._aux.problems[key]
        result = np.empty(lin.data_space.size, complex)
        for frequency in lin.frequencies:
            tables = {}
            for layout in space.term_layouts(frequency=frequency):
                tables[layout.id] = dict(
                    zip(map(tuple, layout.coordinate_keys), observed[layout.indices])
                )
            for layout in lin.data_space.term_layouts(frequency=frequency):
                table = tables[layout.id]
                for index, (source, receiver, component) in zip(
                    layout.indices, layout.coordinate_keys
                ):
                    row_key = (
                        (int(source) - 1) // repeats + 1,
                        int(receiver),
                        int(component),
                    )
                    if row_key not in table:
                        raise ValueError(
                            "aperture receiver rows do not match the original acquisition"
                        )
                    result[index] = table[row_key]
        return result

    def _kernel(self) -> np.ndarray:
        return lag_kernel(self._problem.frequencies, self.focusing.window)

    def _on_space(self, vector: ControlVector) -> ControlVector:
        """Map an auxiliary problem's covector onto this view's space (material blocks)."""

        blocks = vector.space.unpack(vector)
        values = {}
        for block in self.space.resolved_blocks:
            if block.kind in _MATERIAL_KINDS and block.name in blocks:
                values[block.name] = blocks[block.name]
            else:
                values[block.name] = np.zeros(
                    block.size // (2 if block.complex else 1),
                    complex if block.complex else float,
                )
        return self.space.pack(values)

    def _point(self, state: ControlState, gradient: bool) -> FocusingLinearization:
        # Only the receiver state is needed here; the material gradient comes
        # from the focusing dual below, not the baseline L2 objective.
        lin = self._problem.linearize(state, gradient=False)
        J, G, ratio = coherent_focus(
            lin.simulated().values, lin.observed().values, _rows(lin), self._kernel()
        )
        grad = lin.modeled_vjp(G) if gradient else None
        return FocusingLinearization(
            self, state=state, value=J, gradient=grad, ratio=ratio
        )

    # -- auxiliary grid problems ---------------------------------------------

    def _geometry(self) -> Dict[str, Any]:
        aux = self._aux.problems
        if "geometry" not in aux:
            simulation = self._problem.simulation
            points, units, system, kind = _source_points(simulation)
            source, weight = _source_rows(simulation, len(points))
            aperture = self.focusing.aperture
            if aperture is None:
                raise ValueError(
                    "auxiliary focusing geometry requires a source aperture"
                )
            offsets, w, coarse = aperture.nodes(points[source], units)
            interp = _interpolation(offsets, coarse)
            aux["geometry"] = dict(
                points=points,
                units=units,
                system=system,
                kind=kind,
                source=source,
                weight=weight,
                offsets=offsets,
                w=w,
                coarse=coarse,
                interp=interp,
            )
        return aux["geometry"]

    def _aux_problem(
        self,
        tag: str,
        points: np.ndarray,
        encoding: Any,
    ) -> Any:
        """Return (and cache) a problem over this model with another acquisition."""

        from .problem import ImagingProblem

        base = self._problem
        tag = self._stage_key(tag)
        g = self._geometry()
        directory = Path(base.workdir) / "focusing" / tag
        simulation = _swap_sources(
            base.simulation,
            f"{base.simulation.name}__focus_{tag}",
            points,
            g["units"],
            g["system"],
            g["kind"],
            encoding,
            source_indices=np.repeat(g["source"], len(points) // len(g["source"])),
            directory=directory,
        )
        full = base.full_space
        specs = {
            key: spec
            for key, spec in full.specs.items()
            if all(b.kind in _MATERIAL_KINDS for b in full._select(key))
        }
        return ImagingProblem(
            simulation,
            controls=ControlSpace(**specs),
            observed=ObservedData(None),
            misfit=unit_misfit([group.name for group in base.observed_groups]),
            frequencies=list(base.frequencies),
            site=base.site,
            submit_options=base.backend.submit_options,
            workdir=directory,
            name=f"{base.name}_focus_{tag}",
        )

    def _aux_state(self, problem: Any, state: ControlState) -> ControlState:
        """Return ``problem``'s discovered baseline with this state's material blocks."""

        blocks = problem._state_at(None).blocks()
        material = {
            b.name
            for b in self._problem.full_space.resolved_blocks
            if b.kind in _MATERIAL_KINDS
        }
        base = {
            name: value for name, value in state.blocks().items() if name in material
        }
        merged = {
            name: (base[name] if name in base else values)
            for name, values in blocks.items()
        }
        return ControlState.from_blocks(problem.full_space, merged)

    def _extended(self, state: ControlState) -> Any:
        """Linearize sparse extended sources with taper amplitudes."""

        from frequensolve.seismic.sources import SourceEncoding

        aux = self._aux.problems
        key = self._stage_key("extended")
        g = self._geometry()
        if key not in aux:
            R, K = len(g["source"]), len(g["w"])
            nodes = (g["points"][g["source"]][:, None, :] + g["offsets"][None]).reshape(
                -1, g["points"].shape[1]
            )
            encoding = SourceEncoding.named(
                {
                    f"field_{r}": {
                        f"focus_{r * K + k}": float(g["weight"][r] * w)
                        for k, w in enumerate(g["w"])
                    }
                    for r in range(R)
                }
            )
            aux[key] = self._aux_problem("extended", nodes, encoding)
        problem = aux[key]
        return problem.linearize(self._aux_state(problem, state), gradient=False)

    def _linear(self, state: ControlState, gradient: bool) -> FocusingLinearization:
        lin = self._extended(state)
        J, G, ratio = coherent_focus(
            lin.simulated().values, self._observations(lin), _rows(lin), self._kernel()
        )
        grad = self._on_space(lin.modeled_vjp(G)) if gradient else None
        return FocusingLinearization(
            self, state=state, value=J, gradient=grad, ratio=ratio
        )

    def _coarse(self, state: ControlState) -> Any:
        """Linearize sparse independent coarse-node sources with zero observed data."""

        from frequensolve.seismic.sources import SourceEncoding

        aux = self._aux.problems
        key = self._stage_key("coarse")
        g = self._geometry()
        if key not in aux:
            R, Kc = len(g["source"]), len(g["coarse"])
            nodes = (g["points"][g["source"]][:, None, :] + g["coarse"][None]).reshape(
                -1, g["points"].shape[1]
            )
            encoding = SourceEncoding.named(
                {
                    f"field_{i}": {f"focus_{i}": float(g["weight"][i // Kc])}
                    for i in range(R * Kc)
                }
            )
            aux[key] = self._aux_problem("coarse", nodes, encoding)
        problem = aux[key]
        return problem.linearize(self._aux_state(problem, state), gradient=False)

    def _pointwise(self, state: ControlState, gradient: bool) -> FocusingLinearization:
        g = self._geometry()
        R, Kc = len(g["source"]), len(g["coarse"])
        coarse = self._coarse(state)
        dc = coarse.simulated().values
        observed = self._observations(coarse, repeats=Kc)
        # Integrating interpolated ratios is equivalent to this coarse quadrature.
        weights = (g["w"] / g["w"].sum()) @ g["interp"]
        J, G, ratios = coherent_focus(
            dc, observed, _rows(coarse), self._kernel(), np.tile(weights, R)
        )
        grad = self._on_space(coarse.modeled_vjp(G)) if gradient else None
        ratio = ratios.reshape(R, Kc) @ g["interp"].T
        return FocusingLinearization(
            self,
            state=state,
            value=J,
            gradient=grad,
            ratio=ratio,
            details={"coarse_ratio": ratios.reshape(R, Kc)},
        )


def _interpolation(offsets: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    """Return ``a`` (``K x Kc``) interpolating coarse-node values multilinearly."""

    from scipy.interpolate import RegularGridInterpolator

    axes = [np.unique(coarse[:, i]) for i in range(coarse.shape[1])]
    shape = [len(a) for a in axes]
    a = np.zeros((len(offsets), len(coarse)))
    for i, node in enumerate(coarse):
        unit = np.zeros(shape)
        unit[tuple(int(np.argmin(np.abs(ax - x))) for ax, x in zip(axes, node))] = 1.0
        a[:, i] = RegularGridInterpolator(
            axes, unit, bounds_error=False, fill_value=None
        )(offsets)
    return a
