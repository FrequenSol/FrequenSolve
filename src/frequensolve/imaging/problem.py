"""Imaging problems and their linearizations.

:class:`ImagingProblem` binds a simulation, a control space, observed data, a
misfit, a frequency list and an execution site once.  Everything else derives
from it: scalar objective values, gradients, the Jacobian and Gauss-Newton
normal operators of :mod:`frequensolve.imaging.operators`, stage views over
restricted block/frequency subsets, and validation checks.

Every :meth:`ImagingProblem.linearize` maps to one saved Sauce ``fwi_operator``
state (one task per frequency).  Linearizations are cached by a content
fingerprint of the problem identity and the full control state; ``jvp`` /
``vjp`` / ``normal`` actions reuse the saved state and are memoized per input
vector.

Conventions
-----------
- ``J.H @ r`` returns the **real** control covector (real part of the
  Hermitian pairing, Sauce's ``real_interleaved`` packing, no factor 2).
- Per-frequency task outputs are reduced with unit weights (Sauce's own
  unweighted sum).  Frequency weighting is a Sauce preprocess hook
  (:meth:`~frequensolve.imaging.misfit.Preprocess.frequency_weight`) and is
  therefore already contained in the reported totals and covectors.
- ``problem.smoothing`` is applied by Sauce's ``--smooth`` postprocess to
  :attr:`Linearization.gradient` only; the Jacobian and normal operators stay
  exact so adjoint identities hold.
- Support masks (§4.1.1) are taken from the first ``linearize`` of a problem
  view and held fixed for that view; a view created by :meth:`restrict`
  inherits the masks its parent already adopted unless it asks for
  ``support="refresh"`` (stage transitions).
- A view may override the misfit (or just its loss), the smoothing, the
  support threshold and the per-frequency objective weights of the shared
  problem (:meth:`restrict`); overrides are part of the linearization
  fingerprint, so stage views never share cache entries with the base
  problem when they differ.  Frequency weights scale the objective side
  (value, gradient, normal operator); the Jacobian ``J`` stays unweighted.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import itertools
import math
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from frequensolve.imaging._artifacts import (
    ControlRegistryManifest,
    ControlStateFile,
    ControlVectorFile,
    ObjectiveReport,
    SmoothingConfig,
    unqualified_block_name,
)
from frequensolve.imaging._backend import (
    Backend,
    LinearizationCache,
    LinearizationEntry,
    content_fingerprint,
    fingerprint,
    frequency_weights,
    read_manifest,
    read_report,
    read_smoothed_covector,
    read_state_output,
    read_task_objective_vectors,
    reduce_covectors,
    total_value,
)
from frequensolve.imaging.controls import (
    BoundControlSpace,
    ControlSpace,
    ControlState,
    ControlVector,
    SupportMask,
)
from frequensolve.imaging.data import (
    DataSpace,
    DataVector,
    ObservedData,
    ObservedGroup,
    TraceStoreRef,
)
from frequensolve.imaging.jobs import FWIOperatorJob
from frequensolve.imaging.misfit import Loss, Misfit
from frequensolve.imaging.operators import Jacobian, Normal
from frequensolve.inversion.validation import gradient_taylor_test, real_adjoint_test
from frequensolve.orchestrator.sites.base import BaseSite

__all__ = ["ImagingProblem", "Linearization"]

_VectorLike = Union[ControlVector, np.ndarray, Sequence[float]]


class _Unset:
    """Sentinel type for keyword arguments where ``None`` is a valid value."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


_UNSET: Any = _Unset()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _digest(values: Any) -> str:
    """Return a hex digest of an array's bytes (memoization key)."""

    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _normalize_frequencies(values: Iterable[Any]) -> List[Any]:
    """Normalize Laplace samples like :class:`FWIOperatorJob` does."""

    array = np.asarray(list(values))
    if array.size == 0:
        raise ValueError("an imaging problem requires at least one frequency")
    if np.iscomplexobj(array):
        return [complex(value.real, -abs(value.imag)) for value in array]
    return array.astype(float).tolist()


def _project_path(simulation: Any) -> Path:
    project = getattr(simulation, "project_path", None)
    if project is None:
        raise ValueError("the simulation has no project_path; save it in a project")
    return Path(project).expanduser().resolve()


def _resolve_path(value: Any, project: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project / path


def _vector_from_file(file: ControlVectorFile, space: ControlSpace) -> ControlVector:
    """Read a covector file onto ``space`` without adopting its support masks."""

    full = np.zeros(space.full_size, dtype=np.float64)
    for name, sl in space.full_slices.items():
        try:
            values = file[name]
        except KeyError as exc:
            raise ValueError(f"covector has no block {name!r}") from exc
        if values.size != sl.stop - sl.start:
            raise ValueError(
                f"block {name!r} has {values.size} entries in the covector; the "
                f"space has {sl.stop - sl.start}"
            )
        full[sl] = values
    return space.from_sauce_vector(full)


def _observed_descriptor(group: ObservedGroup) -> Dict[str, Any]:
    """Describe one observed group for fingerprinting (content hashes when local)."""

    payload: Dict[str, Any] = group.to_fs()
    hashes: Dict[str, Any] = {}
    refs: List[Tuple[str, Any]] = []
    if group.observed is not None:
        refs.append(("observed", group.observed))
    for axis, ref in group.derivatives.items():
        refs.append((f"derivative:{axis}", ref))
    for label, ref in refs:
        path = Path(ref.file) if isinstance(ref, TraceStoreRef) else Path(ref)
        if path.exists():
            hashes[label] = content_fingerprint(path)
    if hashes:
        payload["content"] = hashes
    return payload


def _misfit_with_loss(misfit: Misfit, loss: Any) -> Misfit:
    """Return ``misfit`` with every objective term's loss replaced by ``loss``."""

    loss = Loss.from_value(loss)
    explicit = misfit.explicit_terms
    if explicit is not None:
        terms = [dataclasses.replace(term, loss=loss) for term in explicit]
        return Misfit(
            terms=terms,
            preprocess=misfit.preprocess,
            projection=misfit.projection,
            group_preprocess=misfit.group_preprocess,
            include_default_preprocess=misfit.include_default_preprocess,
        )
    return Misfit(
        loss=loss,
        comparison=misfit.comparison,
        normalization=misfit.normalization,
        weights=misfit.weights,
        preprocess=misfit.preprocess,
        projection=misfit.projection,
        group_preprocess=misfit.group_preprocess,
        include_default_preprocess=misfit.include_default_preprocess,
    )


class _MisfitPayload:
    """``Imaging.misfit`` adapter with the job's ``to_fs(ctx)`` calling convention."""

    def __init__(self, misfit: Misfit, groups: Sequence[ObservedGroup]) -> None:
        self.misfit = misfit
        self.groups = list(groups)

    def to_fs(
        self, ctx: Any = None, *, project_relative: bool = False
    ) -> Dict[str, Any]:
        return self.misfit.to_fs(self.groups, ctx)

    def __repr__(self) -> str:
        return f"_MisfitPayload({self.misfit!r})"


# ---------------------------------------------------------------------------
# shared problem state
# ---------------------------------------------------------------------------


class _Shared:
    """State shared by an :class:`ImagingProblem` and its restricted views."""

    def __init__(
        self,
        simulation: Any,
        *,
        controls: Any,
        observed: Any,
        misfit: Misfit,
        frequencies: Optional[Iterable[Any]],
        site: Optional[BaseSite],
        smoothing: Any,
        workdir: Optional[Union[str, Path]],
        name: str,
        min_support: Optional[float],
        submit_options: Optional[Mapping[str, Any]],
        cache_capacity: int,
    ) -> None:
        self.name = str(name).strip()
        if not self.name or "/" in self.name:
            raise ValueError("ImagingProblem name must be a non-empty path segment")
        if not isinstance(controls, ControlSpace):
            controls = ControlSpace(controls)
        self.project = _project_path(simulation)
        # Jobs persist their simulation under ``<project>/simulations/<name>``;
        # work on a renamed copy so the authored simulation file stays untouched.
        working = simulation.copy(f"{simulation.name}__{self.name}")
        self.space: BoundControlSpace = controls.bind(working)
        self.simulation = self.space.simulation
        self.observed_source = observed
        self.observed_data = (
            observed if isinstance(observed, ObservedData) else ObservedData(observed)
        )
        self.groups: List[ObservedGroup] = [
            group.resolved(lambda p: _resolve_path(p, self.project))
            for group in self.observed_data.resolve(self.simulation)
        ]
        if not isinstance(misfit, Misfit):
            raise TypeError("misfit must be a Misfit")
        self.misfit = misfit
        self.misfit_payload = _MisfitPayload(misfit, self.groups)
        if frequencies is None:
            frequencies = self.observed_data.frequencies
            if frequencies is None:
                raise ValueError(
                    "frequencies are required when the observed data does not "
                    "declare them"
                )
        self.frequencies: List[Any] = _normalize_frequencies(frequencies)
        self.smoothing: Optional[SmoothingConfig] = SmoothingConfig.from_value(
            smoothing
        )
        if min_support is not None:
            threshold = float(min_support)
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError("min_support must be finite and non-negative")
            min_support = threshold
        self.min_support = min_support
        self.workdir = (
            Path(workdir).expanduser().resolve()
            if workdir is not None
            else self.project / "imaging" / self.name
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._site = site
        self._backend: Optional[Backend] = None
        self.submit_options: Dict[str, Any] = dict(submit_options or {})
        self.cache = LinearizationCache(self.workdir, cache_capacity)
        self.linearizations: Dict[str, "Linearization"] = {}
        self.pending_manifest = not self.space.resolved
        self.authored: Optional[ControlState] = (
            None if self.pending_manifest else ControlState.from_simulation(self.space)
        )
        self.state: Optional[ControlState] = self.authored
        # Sauce's ``controls.state`` must carry every registry block (source
        # blocks included); the complete baseline is learned from the first
        # ``state_output`` and merged with the space's blocks afterwards.
        self.baseline: Optional[ControlStateFile] = None
        self._identity: Optional[Dict[str, Any]] = None

    @property
    def site(self) -> BaseSite:
        if self._site is None:
            from frequensolve.orchestrator.sites.config_file import Site

            self._site = Site()
        return self._site

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = Backend(
                self.site,
                self.workdir,
                submit_options=self.submit_options,
                prefix=self.name,
            )
        return self._backend

    def identity(self) -> Dict[str, Any]:
        """Return the state-independent fingerprint parts (computed once)."""

        if self._identity is None:
            try:
                simulation: Any = self.simulation.to_fs()
            except Exception:  # pragma: no cover - defensive fallback
                simulation = repr(self.simulation)
            try:
                misfit: Any = self.misfit_payload.to_fs()
            except ValueError:  # large trace weights need an export context
                misfit = repr(self.misfit)
            self._identity = {
                "problem": self.name,
                "simulation": simulation,
                "misfit": misfit,
                "observed": [_observed_descriptor(group) for group in self.groups],
                "min_support": self.min_support,
                "smoothing": (
                    None if self.smoothing is None else self.smoothing.to_control_fs()
                ),
            }
        return self._identity

    def set_state(self, state: ControlState) -> None:
        if not isinstance(state, ControlState):
            raise TypeError("problem.state must be a ControlState")
        if self.pending_manifest:
            raise ValueError(
                "the control layout is unknown until the first linearize supplies "
                "the registry manifest"
            )
        if (
            state.space.blocks != self.space.blocks
            or state.size != self.space.full_size
        ):
            raise ValueError("state does not cover the problem's control blocks")
        self.state = ControlState(self.space.without_support(), state.values)

    def forget(self, entries: Iterable[LinearizationEntry]) -> None:
        for entry in entries:
            self.linearizations.pop(entry.fingerprint, None)


# ---------------------------------------------------------------------------
# problem
# ---------------------------------------------------------------------------


class ImagingProblem:
    """Declare an imaging inverse problem once and derive everything from it.

    Args:
        simulation: Simulation whose model and acquisition define the control
            registry.  It is deep-copied; the caller's object is untouched.
        controls: :class:`~frequensolve.imaging.controls.ControlSpace` or a
            single block.
        observed: :class:`~frequensolve.imaging.data.ObservedData` or any
            source it accepts (forward job, path stem, mapping, dataset).
        misfit: :class:`~frequensolve.imaging.misfit.Misfit`; defaults to
            ``Misfit.l2()``.
        frequencies: Frequencies solved per linearization; inferred from the
            observed data when omitted.
        site: Execution site; defaults to the configured ``Site()`` lazily.
        smoothing: Optional :class:`SmoothingConfig` (or mapping) applied by
            Sauce's ``--smooth`` postprocess to linearize gradients.
        workdir: Directory for staged inputs and cache bookkeeping; defaults
            to ``<project>/imaging/<name>``.
        name: Problem name, used for job prefixes and the default workdir.
        min_support: ``fwi_operator.controls.min_support`` relative support
            threshold (Sauce default ``1e-2``).
        submit_options: Site submission options pinned for every job.
        cache_capacity: Number of linearizations kept alive.
    """

    def __init__(
        self,
        simulation: Any,
        *,
        controls: Any,
        observed: Any,
        misfit: Optional[Misfit] = None,
        frequencies: Optional[Iterable[Any]] = None,
        site: Optional[BaseSite] = None,
        smoothing: Any = None,
        workdir: Optional[Union[str, Path]] = None,
        name: str = "imaging",
        min_support: Optional[float] = None,
        submit_options: Optional[Mapping[str, Any]] = None,
        cache_capacity: int = 2,
    ) -> None:
        self._shared = _Shared(
            simulation,
            controls=controls,
            observed=observed,
            misfit=Misfit.l2() if misfit is None else misfit,
            frequencies=frequencies,
            site=site,
            smoothing=smoothing,
            workdir=workdir,
            name=name,
            min_support=min_support,
            submit_options=submit_options,
            cache_capacity=cache_capacity,
        )
        self._active: Optional[Tuple[str, ...]] = None
        self._frequencies: Tuple[Any, ...] = tuple(self._shared.frequencies)
        self._masks: Dict[str, np.ndarray] = {}
        self._masks_adopted = False
        self._data_space: Optional[DataSpace] = None
        # View-level overrides of shared settings (``restrict``); absent keys
        # fall through to ``_shared``.
        self._overrides: Dict[str, Any] = {}
        self._misfit_payload_cache: Optional[_MisfitPayload] = None
        report = self.capabilities()
        if report["errors"]:
            raise ValueError(
                "imaging problem is not supported by Sauce: "
                + "; ".join(report["errors"])
            )

    # -- identity -------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._shared.name

    @property
    def simulation(self) -> Any:
        """Return the bound copy of the simulation (controls installed)."""

        return self._shared.simulation

    @property
    def site(self) -> BaseSite:
        return self._shared.site

    @property
    def backend(self) -> Backend:
        return self._shared.backend

    @property
    def workdir(self) -> Path:
        return self._shared.workdir

    @property
    def cache(self) -> LinearizationCache:
        return self._shared.cache

    @property
    def observed_data(self) -> ObservedData:
        return self._shared.observed_data

    @property
    def observed_groups(self) -> List[ObservedGroup]:
        """Return the observed groups resolved against the simulation."""

        return list(self._shared.groups)

    @property
    def misfit(self) -> Misfit:
        """Return this view's misfit (a :meth:`restrict` override or the shared one)."""

        misfit = self._overrides.get("misfit")
        return self._shared.misfit if misfit is None else misfit

    @property
    def _misfit_payload(self) -> _MisfitPayload:
        if "misfit" not in self._overrides:
            return self._shared.misfit_payload
        if self._misfit_payload_cache is None:
            self._misfit_payload_cache = _MisfitPayload(
                self.misfit, self._shared.groups
            )
        return self._misfit_payload_cache

    @property
    def smoothing(self) -> Optional[SmoothingConfig]:
        """Return this view's gradient smoothing (override or shared)."""

        if "smoothing" in self._overrides:
            return self._overrides["smoothing"]
        return self._shared.smoothing

    @property
    def min_support(self) -> Optional[float]:
        """Return this view's relative support threshold (override or shared)."""

        if "min_support" in self._overrides:
            return self._overrides["min_support"]
        return self._shared.min_support

    @property
    def weights(self) -> Optional[Tuple[float, ...]]:
        """Return this view's per-frequency objective weights (``None`` = unit)."""

        return self._overrides.get("weights")

    def identity(self) -> Dict[str, Any]:
        """Return the state-independent identity of this view.

        The mapping combines the shared problem identity (simulation, misfit,
        observed data, smoothing, support threshold) with this view's active
        blocks, frequencies and overrides.  Workflows fingerprint it to reject
        checkpoints that belong to a different problem.
        """

        return {
            **self._shared.identity(),
            "active": list(self.space.blocks),
            "frequencies": list(self._frequencies),
            **self._override_identity(),
        }

    def _override_identity(self) -> Dict[str, Any]:
        """Return the fingerprint contribution of this view's overrides."""

        parts: Dict[str, Any] = {}
        if "misfit" in self._overrides:
            try:
                parts["misfit_override"] = self._misfit_payload.to_fs()
            except ValueError:  # large trace weights need an export context
                parts["misfit_override"] = repr(self.misfit)
        if "smoothing" in self._overrides:
            smoothing = self._overrides["smoothing"]
            parts["smoothing_override"] = (
                None if smoothing is None else smoothing.to_control_fs()
            )
        if "min_support" in self._overrides:
            parts["min_support_override"] = self._overrides["min_support"]
        if "weights" in self._overrides:
            parts["weights_override"] = list(self._overrides["weights"])
        return parts

    @property
    def frequencies(self) -> List[Any]:
        """Return the frequencies of this view."""

        return list(self._frequencies)

    @property
    def full_space(self) -> BoundControlSpace:
        """Return the complete bound space (every block, no support masks)."""

        return self._shared.space

    @property
    def space(self) -> ControlSpace:
        """Return this view's control space (active blocks, adopted support)."""

        space: ControlSpace = self._shared.space
        if self._active is not None:
            space = space.restrict(list(self._active))
        if self._masks:
            space = space.with_support(self._masks, self.min_support)
        return space

    @property
    def data_space(self) -> DataSpace:
        if self._data_space is None:
            self._data_space = DataSpace.from_simulation(
                self.simulation, frequencies=self._frequencies
            )
        return self._data_space

    @property
    def state(self) -> Optional[ControlState]:
        """Return the current full baseline (shared by every view)."""

        return self._shared.state

    @state.setter
    def state(self, value: ControlState) -> None:
        self._shared.set_state(value)

    def __repr__(self) -> str:
        return (
            f"ImagingProblem(name={self.name!r}, blocks={list(self.space.blocks)}, "
            f"frequencies={self.frequencies})"
        )

    # -- views ----------------------------------------------------------------

    def restrict(
        self,
        frequencies: Optional[Iterable[Any]] = None,
        active: Optional[Union[str, Sequence[str]]] = None,
        *,
        misfit: Optional[Misfit] = None,
        loss: Any = None,
        smoothing: Any = _UNSET,
        min_support: Any = _UNSET,
        weights: Any = _UNSET,
        support: str = "inherit",
    ) -> "ImagingProblem":
        """Return a stage view sharing state, cache, backend and simulation.

        Args:
            frequencies: Subset of this view's frequencies (matched with
                ``numpy.isclose``).
            active: Block keys, addresses or qualified names selecting the
                active subspace, in order.
            misfit: Replace the misfit for this view (stage-level objective).
            loss: Keep this view's misfit but swap the loss of every term
                (``"huber"``, :class:`~frequensolve.imaging.misfit.Loss`, ...).
                Cannot be combined with ``misfit``.
            smoothing: Replace the gradient smoothing for this view
                (``None`` switches it off; omit to inherit).
            min_support: Replace the relative support threshold for this
                view (``None`` restores Sauce's default; omit to inherit).
            weights: One nonnegative objective weight per frequency of the
                new view (``None`` restores unit weights; omit to inherit,
                in which case an inherited table is sliced with
                ``frequencies``).  Weights scale value, gradient and normal
                operator; ``J`` stays unweighted.
            support: ``"inherit"`` reuses the support masks this view already
                adopted; ``"refresh"`` makes the new view adopt masks from its
                own first ``linearize`` (stage transitions, spec §4.1.1).
        """

        if support not in {"inherit", "refresh"}:
            raise ValueError("support must be 'inherit' or 'refresh'")
        if misfit is not None and loss is not None:
            raise ValueError("pass either misfit or loss, not both")
        view = object.__new__(ImagingProblem)
        view._shared = self._shared
        view._data_space = None
        view._overrides = dict(self._overrides)
        view._misfit_payload_cache = None
        if misfit is not None:
            if not isinstance(misfit, Misfit):
                raise TypeError("misfit must be a Misfit")
            view._overrides["misfit"] = misfit
        elif loss is not None:
            view._overrides["misfit"] = _misfit_with_loss(self.misfit, loss)
        if smoothing is not _UNSET:
            view._overrides["smoothing"] = SmoothingConfig.from_value(smoothing)
        if min_support is not _UNSET:
            if min_support is not None:
                threshold = float(min_support)
                if not math.isfinite(threshold) or threshold < 0.0:
                    raise ValueError("min_support must be finite and non-negative")
                min_support = threshold
            view._overrides["min_support"] = min_support
        if active is None:
            view._active = self._active
        else:
            view._active = tuple(self.space.restrict(active).blocks)
        inherited = self._overrides.get("weights")
        if frequencies is None:
            view._frequencies = self._frequencies
        else:
            selected: List[Any] = []
            positions: List[int] = []
            available = np.asarray(self._frequencies, dtype=complex)
            for value in _normalize_frequencies(frequencies):
                matches = np.flatnonzero(np.isclose(available, complex(value)))
                if matches.size != 1:
                    raise ValueError(
                        f"frequency {value!r} is not one of {self.frequencies}"
                    )
                canonical = self._frequencies[int(matches[0])]
                if canonical in selected:
                    raise ValueError(f"frequency {value!r} selected twice")
                selected.append(canonical)
                positions.append(int(matches[0]))
            view._frequencies = tuple(selected)
            if inherited is not None:
                view._overrides["weights"] = tuple(inherited[i] for i in positions)
        if weights is not _UNSET:
            if weights is None:
                view._overrides.pop("weights", None)
            else:
                table = np.asarray(weights, dtype=np.float64).reshape(-1)
                if table.size != len(view._frequencies):
                    raise ValueError(
                        f"expected {len(view._frequencies)} frequency weights, "
                        f"received {table.size}"
                    )
                if not np.all(np.isfinite(table)) or np.any(table < 0.0):
                    raise ValueError("frequency weights must be finite and nonnegative")
                view._overrides["weights"] = tuple(float(w) for w in table)
        names = set(view._active or self._shared.space.blocks)
        if support == "refresh":
            view._masks = {}
            view._masks_adopted = False
        else:
            view._masks = {n: m for n, m in self._masks.items() if n in names}
            view._masks_adopted = self._masks_adopted
        report = view.capabilities()
        if report["errors"]:
            raise ValueError(
                "restricted imaging problem is not supported by Sauce: "
                + "; ".join(report["errors"])
            )
        return view

    def with_misfit(
        self, misfit: Optional[Misfit] = None, *, loss: Any = None
    ) -> "ImagingProblem":
        """Return this view with another misfit (or loss); see :meth:`restrict`."""

        return self.restrict(misfit=misfit, loss=loss)

    # -- states and vectors ---------------------------------------------------

    def _require_state(self) -> ControlState:
        state = self._shared.state
        if state is None:
            raise ValueError(
                "the control state is unknown until the first linearize supplies "
                "the registry manifest (mesh controls)"
            )
        return state

    def _state_at(self, v: Any, base: Optional[ControlState] = None) -> ControlState:
        """Return the full state at ``v`` (vector on this view's space)."""

        if isinstance(v, ControlState):
            if v.space.blocks != self._shared.space.blocks:
                raise ValueError("state does not cover the problem's control blocks")
            return v
        state = self._require_state() if base is None else base
        if v is None:
            return state
        vector = v if isinstance(v, ControlVector) else ControlVector(v, self.space)
        return state.with_update(vector)

    def vector(self, state: Optional[ControlState] = None) -> ControlVector:
        """Return the active slice of ``state`` (default: the current state)."""

        return (self._require_state() if state is None else state).vector(self.space)

    def state_from(self, vector: _VectorLike) -> ControlState:
        """Return the current state with ``vector`` written into the active DOFs."""

        return self._state_at(vector)

    # -- linearization --------------------------------------------------------

    def _fingerprint(self, state: Optional[ControlState]) -> str:
        return fingerprint(
            **self.identity(),
            state=None if state is None else state.values,
        )

    def _control_active(self, space: ControlSpace) -> Optional[List[str]]:
        """Return ``control_sensitivities.active`` for a smoothed job."""

        model = [
            unqualified_block_name(name)
            for name in space.blocks
            if name.startswith("model.")
        ]
        if not model:
            raise ValueError("smoothing requires at least one model.* block")
        return None if len(model) == len(space.blocks) else model

    def _linearize_job(
        self,
        space: ControlSpace,
        control_state: Optional[Path],
        *,
        gradient: bool,
    ) -> FWIOperatorJob:
        shared = self._shared
        smoothing = self.smoothing if gradient else None
        return FWIOperatorJob(
            shared.backend.job_name("linearize"),
            shared.simulation,
            list(self._frequencies),
            action="linearize",
            active=list(space.blocks),
            state="state.json",
            covector="gradient.h5" if gradient else None,
            objective="report.json",
            control_state=control_state,
            state_output="baseline.h5",
            manifest="registry.json",
            min_support=self.min_support,
            misfit=self._misfit_payload,
            reflectivity=self._reflectivity(space),
            source_controls=self._source_controls(space),
            # Sauce only accepts weights on covector-carrying jobs (they drive
            # the --smooth aggregation); value-only linearizations weight the
            # reports on the FrequenSolve side (Linearization.frequency_weights).
            weights=(
                None if self.weights is None or not gradient else list(self.weights)
            ),
            smoothing=smoothing,
            control_active=(
                self._control_active(space) if smoothing is not None else None
            ),
        )

    def _run_job(self, job: FWIOperatorJob) -> None:
        """Run ``job`` through the backend and fail on any failed task."""

        self.backend.run(job)
        self._check_tasks(job)

    def _run_jobs(self, jobs: Sequence[FWIOperatorJob]) -> None:
        """Run a job family (siblings submitted before waiting) and check tasks."""

        self.backend.run_many(jobs)
        for job in jobs:
            self._check_tasks(job)

    @staticmethod
    def _check_tasks(job: FWIOperatorJob) -> None:
        """Fail on any failed task of a finished job.

        Sites may report a run as completed while a task failed on solver
        convergence; surface those failures with their reasons.
        """

        failed_tasks = getattr(job, "failed_tasks", None)
        failures = failed_tasks() if callable(failed_tasks) else []
        if failures:
            details = "; ".join(
                f"task {row.get('task')} ({row.get('frequency')} Hz): "
                f"{row.get('reason') or row.get('status')}"
                for row in failures
            )
            raise RuntimeError(
                f"{job.action} job {job.name!r} has failed task(s): {details}; "
                f"see {job._stdout_path}"
            )

    def _source_controls(self, space: ControlSpace) -> Optional[Dict[str, Any]]:
        if any(b.kind == "source" for b in space.resolved_blocks):
            return {"location_method": "analytic"}
        return None

    def _reflectivity(self, space: ControlSpace) -> Optional[Dict[str, Any]]:
        """Return ``fwi_operator.reflectivity`` for the active blocks, if any."""

        if not any(b.kind == "reflectivity" for b in space.resolved_blocks):
            return None
        restricted = self._shared.space.restrict(list(space.blocks))
        payload = getattr(restricted, "reflectivity_payload", None)
        return None if payload is None else payload()

    def _operator_job(
        self,
        space: ControlSpace,
        action: str,
        *,
        frequencies: Sequence[Any],
        **kwargs: Any,
    ) -> FWIOperatorJob:
        """Build one jvp/vjp/normal job over ``space`` (shared by Linearization)."""

        shared = self._shared
        return FWIOperatorJob(
            shared.backend.job_name(action),
            shared.simulation,
            list(frequencies),
            action=action,
            active=list(space.blocks),
            misfit=self._misfit_payload,
            reflectivity=self._reflectivity(space),
            source_controls=self._source_controls(space),
            **kwargs,
        )

    def _is_authored(self, state: Optional[ControlState]) -> bool:
        authored = self._shared.authored
        return (
            state is None
            or authored is None
            or np.array_equal(state.values, authored.values)
        )

    def is_authored(self, state: Optional[ControlState] = None) -> bool:
        """Return whether ``state`` (default: the current one) is the authored baseline."""

        return self._is_authored(self._shared.state if state is None else state)

    def _stage_state(
        self, key: str, state: Optional[ControlState]
    ) -> Tuple[Path, Optional[Path]]:
        """Stage ``controls.state`` for ``state``; ``None`` runs the authored baseline.

        Sauce requires the complete registry baseline, so a state can only be
        written once :attr:`_Shared.baseline` is known (after the first
        linearize).  The authored baseline itself needs no file.
        """

        stage = self.workdir / key.split(":")[-1][:16]
        stage.mkdir(parents=True, exist_ok=True)
        baseline = self._shared.baseline
        if state is None or self._is_authored(state) or baseline is None:
            return stage, None
        blocks = dict(baseline.blocks)
        for name, values in state.blocks().items():
            if name not in blocks:
                raise ValueError(f"Sauce's control registry has no block {name!r}")
            if values.size != blocks[name].size:
                raise ValueError(
                    f"block {name!r} has {values.size} DOFs locally but "
                    f"{blocks[name].size} in Sauce's registry"
                )
            blocks[name] = values
        path = stage / "state.h5"
        ControlStateFile(blocks).write(path)
        return stage, path

    def _read_masks(
        self, job: FWIOperatorJob, space: ControlSpace
    ) -> Optional[Dict[str, np.ndarray]]:
        """Return Sauce's support masks for ``space`` or ``None`` when absent."""

        file: Any = None
        if job.state_output is not None and job.state_output_file().is_file():
            file = read_state_output(job)
        elif job.covector is not None and job.covector_file(1).is_file():
            file = ControlVectorFile.read(job.covector_file(1), native=False)
        if file is None or not file.support:
            return None
        return {
            name: file.support_mask(name)
            for name in space.blocks
            if name in file.blocks
        }

    def _fallback_masks(self, space: ControlSpace) -> Dict[str, np.ndarray]:
        bound = self._shared.space
        if not isinstance(bound, BoundControlSpace):
            return {}
        try:
            masks = bound.geometric_support()
        except Exception:  # pragma: no cover - geometry fallback is best effort
            return {}
        return {name: mask for name, mask in masks.items() if name in space.blocks}

    def _adopt_masks(self, masks: Mapping[str, np.ndarray]) -> None:
        if self._masks_adopted:
            return
        self._masks = {
            name: np.array(mask, dtype=bool)
            for name, mask in masks.items()
            if not np.all(mask)
        }
        self._masks_adopted = True

    def _linearize_state(
        self, state: Optional[ControlState], *, gradient: bool
    ) -> "Linearization":
        shared = self._shared
        key = self._fingerprint(state)
        cached = shared.linearizations.get(key)
        if cached is not None and (cached.gradient is not None or not gradient):
            shared.cache.get(key)
            return cached
        if shared.baseline is None and not self._is_authored(state):
            # Discover the complete registry baseline at the authored point.
            self._linearize_state(shared.authored, gradient=False)
        space = self.space
        stage, control_state = self._stage_state(key, state)
        job = self._linearize_job(space, control_state, gradient=gradient)
        self._run_job(job)

        manifest = read_manifest(job)
        if job.state_output is not None and job.state_output_file().is_file():
            shared.baseline = read_state_output(job)
        if shared.pending_manifest:
            shared.space = shared.space.with_manifest(manifest)  # type: ignore[assignment]
            shared.pending_manifest = False
            shared.state = ControlState.from_file(
                read_state_output(job), shared.space.without_support()
            )
            state = shared.state
            space = self.space
        assert state is not None

        masks = self._read_masks(job, space)
        if masks is None:
            masks = self._fallback_masks(space)
        self._adopt_masks(masks)
        space = self.space

        reports = read_report(job)
        state_fp: Optional[str] = None
        registry_fp: Optional[str] = manifest.fingerprint
        if job.covector is not None and job.covector_file(1).is_file():
            part = ControlVectorFile.read(job.covector_file(1), native=False)
            state_fp = part.state_fingerprint
            registry_fp = part.control_registry_fingerprint or registry_fp
        if state_fp is None:
            state_fp = next(
                (r.state_fingerprint for r in reports if r.state_fingerprint), None
            )
        if state_fp is None or registry_fp is None:
            raise ValueError(
                f"linearize job {job.name!r} reported no state/registry fingerprints"
            )
        entry = LinearizationEntry(
            fingerprint=key,
            job=job,
            directory=stage,
            state_fingerprint=state_fp,
            control_registry_fingerprint=registry_fp,
            extra={"active": list(space.blocks), "frequencies": list(job.f_list)},
        )
        gradient_file: Optional[ControlVectorFile] = None
        if gradient:
            gradient_file = (
                read_smoothed_covector(job)
                if job.requires_postprocess()
                else reduce_covectors(job)
            )
        linearization = Linearization(
            self,
            space=space,
            state=state,
            entry=entry,
            manifest=manifest,
            reports=reports,
            gradient_file=gradient_file,
            support_masks=masks,
        )
        shared.forget(shared.cache.put(entry))
        shared.linearizations[key] = linearization
        return linearization

    def linearize(self, v: Any = None, *, gradient: bool = True) -> "Linearization":
        """Save (or reuse) the Sauce state at ``v`` and return its linearization.

        Args:
            v: Point on this view's space (``ControlVector`` or array), a full
                ``ControlState``, or ``None`` for the current state.
            gradient: Request the covector.  A cached gradient-carrying
                linearization of the same point is reused either way.

        The first linearize of a problem runs at the authored baseline and
        learns Sauce's complete registry baseline from ``state_output``; a
        non-authored point requested before that triggers one value-only
        linearize at the authored baseline first.
        """

        state = (
            None if (v is None and self._shared.state is None) else self._state_at(v)
        )
        return self._linearize_state(state, gradient=gradient)

    def value(self, v: Any = None) -> float:
        """Return the misfit value at ``v``."""

        return self.linearize(v, gradient=False).value

    def gradient(self, v: Any = None) -> ControlVector:
        """Return the misfit gradient (real covector) at ``v``."""

        gradient = self.linearize(v, gradient=True).gradient
        assert gradient is not None
        return gradient

    def jacobian(self, v: Any = None) -> Jacobian:
        """Return the Jacobian operator at ``v``."""

        return self.linearize(v, gradient=False).jacobian

    def normal(self, v: Any = None) -> Normal:
        """Return the Gauss-Newton normal operator at ``v``."""

        return self.linearize(v, gradient=False).normal

    # -- forward modelling ----------------------------------------------------

    def forward(self, v: Any = None) -> DataVector:
        """Return the synthetic data at ``v`` over this view's frequencies.

        Sites exposing a ``forward(simulation, state_file, frequencies,
        active=...)`` hook (the test fake) are used directly; otherwise a
        :class:`~frequensolve.simulation.jobs.forward.FrequencyDomainJob` runs
        on :meth:`simulation_at` and its traces are packed.
        """

        state = self._state_at(v)
        hook = getattr(self.site, "forward", None)
        if callable(hook):
            data = hook(
                self.simulation,
                ControlStateFile(state.blocks()),
                list(self._frequencies),
                active=list(self.space.blocks),
            )
            return DataVector(self.data_space.pack(data), self.data_space)
        from frequensolve.simulation.jobs.forward import FrequencyDomainJob

        job = FrequencyDomainJob(
            self.backend.job_name("forward"),
            self.simulation_at(state),
            list(self._frequencies),
        )
        result = self.backend.run(job)
        return DataVector(self.data_space.pack(result.traces()), self.data_space)

    def observed_vector(self) -> DataVector:
        """Return the observed data as a :class:`DataVector`.

        Available when the site exposes an ``observed`` hook (the test fake),
        or when the observed data came from a forward job or a
        ``TraceDataset``; packed trace stores need a reader that arrives with
        the workflow layer.
        """

        hook = getattr(self.site, "observed", None)
        space = self.space
        if callable(hook):
            data = hook(
                self.simulation,
                list(space.blocks),
                list(self._frequencies),
                sizes=space.sizes,
            )
            return DataVector(self.data_space.pack(data), self.data_space)
        source = self._shared.observed_source
        if hasattr(source, "trace_path") and hasattr(source, "f_list"):
            from frequensolve.seismic.traces import TraceDataset

            return DataVector(
                self.data_space.pack(TraceDataset.from_job(source)), self.data_space
            )
        try:
            from frequensolve.seismic.traces import TraceDataset
        except ImportError:  # pragma: no cover - seismic extras present in tests
            TraceDataset = None  # type: ignore[assignment,misc]
        if TraceDataset is not None and isinstance(source, TraceDataset):
            return DataVector(self.data_space.pack(source), self.data_space)
        raise NotImplementedError(
            "observed_vector needs the observed data as a forward job or a "
            "TraceDataset; packed observed trace stores are read by Sauce only"
        )

    def residual(self, v: Any = None) -> DataVector:
        """Return ``forward(v) - observed``."""

        return self.forward(v) - self.observed_vector()

    def simulation_at(self, v: Any = None) -> Any:
        """Return a simulation copy with the state's coefficients installed.

        Material (profile, lattice) and interface blocks are installed; mesh,
        source and reflectivity baselines are not yet supported.
        """

        state = self._state_at(v)
        simulation = copy.deepcopy(self.simulation)
        full = self._shared.space
        for block, sl in zip(full.resolved_blocks, full.full_slices.values()):
            values = np.array(state.values[sl], copy=True)
            block_id = unqualified_block_name(block.name)
            if block.kind in {"profile", "grid"}:
                _install_material(simulation, block_id, values)
            elif block.kind == "interface":
                _install_interface(simulation, block_id, values)
            elif block.kind == "mesh":
                raise NotImplementedError(
                    f"mesh block {block.name!r}: installing mesh control baselines "
                    "requires a control checkpoint (Phase 2)"
                )
            else:
                raise NotImplementedError(
                    f"{block.kind} block {block.name!r}: installing source and "
                    "reflectivity baselines arrives with Phase 3"
                )
        return simulation

    # -- diagnostics ----------------------------------------------------------

    def dry_run(self, v: Any = None) -> Dict[str, Any]:
        """Describe the linearize job at ``v`` without submitting it.

        Before the first linearize the complete registry baseline is unknown,
        so a non-authored ``v`` is described without ``controls.state`` and
        ``payload["registry_discovery"]`` is ``True``.
        """

        state = (
            None if (v is None and self._shared.state is None) else self._state_at(v)
        )
        key = self._fingerprint(state)
        _stage, control_state = self._stage_state(key, state)
        job = self._linearize_job(self.space, control_state, gradient=True)
        payload = self.backend.dry_run(job)
        payload["fingerprint"] = key
        payload["registry_discovery"] = bool(
            self._shared.baseline is None and not self._is_authored(state)
        )
        return payload

    def capabilities(self) -> Dict[str, Any]:
        """Statically validate the block/misfit/physics combination.

        Returns a mapping with ``errors`` (combinations Sauce rejects) and
        ``warnings`` (documented limitations); ``ok`` is ``not errors``.
        """

        shared = self._shared
        space = self.space
        blocks = list(space.resolved_blocks)
        kinds = {block.kind for block in blocks}
        errors: List[str] = []
        warnings: List[str] = []
        misfit = self.misfit
        smoothing = self.smoothing
        comparisons = {
            term.comparison.kind
            for term in misfit.objective_terms([group.name for group in shared.groups])
        }
        if "source" in kinds and comparisons - {"waveform"}:
            errors.append(
                "source controls require waveform comparisons "
                f"(misfit uses {sorted(comparisons)})"
            )
        if smoothing is not None:
            if not any(b.name.startswith("model.") for b in blocks):
                errors.append("smoothing requires at least one model.* block")
            if "grid" in kinds:
                warnings.append(
                    "tensor_hat (GridParameters) smoothing is an unimplemented "
                    "Sauce stub"
                )
        if shared.pending_manifest:
            warnings.append(
                "mesh control sizes are unknown until the first linearize returns "
                "the registry manifest"
            )
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "blocks": list(space.blocks),
            "kinds": sorted(kinds),
            "comparisons": sorted(comparisons),
            "frequencies": self.frequencies,
            "smoothing": None if smoothing is None else smoothing.kind,
        }

    def clear_cache(self) -> None:
        """Forget every linearization and remove owned staging directories."""

        self._shared.forget(self._shared.cache.clear())
        self._shared.linearizations.clear()

    def check(
        self,
        v: Any = None,
        *,
        seed: int = 0,
        steps: Sequence[float] = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3),
        tolerance: float = 1.0e-8,
        taylor: bool = True,
    ) -> Dict[str, Any]:
        """Run adjoint, normal-consistency and Taylor tests at ``v``.

        Returns a mapping with ``adjoint`` (``<J dv, r>_Re`` versus
        ``<dv, J^H r>``), ``normal`` (``H dv`` versus ``J^H W J dv``), optionally
        ``taylor`` (:func:`~frequensolve.inversion.validation.gradient_taylor_test`
        on ``value``/``gradient``), and ``passed``.
        """

        lin = self.linearize(v, gradient=taylor)
        J = lin.jacobian
        dv = lin.space.random(seed)
        r = lin.data_space.random(seed + 1)
        adjoint = real_adjoint_test(
            lambda p: np.asarray(J @ p),
            lambda y: np.asarray(J.H @ y),
            dv.values,
            r.values,
            relative_tolerance=tolerance,
        )
        h_dv = np.asarray(lin.normal @ dv)
        jhj_dv = np.asarray(J.H @ lin.weight_data(J @ dv))
        scale = max(float(np.linalg.norm(jhj_dv)), float(np.finfo(np.float64).tiny))
        normal_error = float(np.linalg.norm(h_dv - jhj_dv) / scale)
        normal = {"relative_error": normal_error, "passed": normal_error <= tolerance}
        report: Dict[str, Any] = {
            "fingerprint": lin.fingerprint,
            "adjoint": adjoint,
            "normal": normal,
        }
        passed = bool(adjoint["passed"] and normal["passed"])
        if taylor:
            base = lin.state
            space = lin.space

            def objective(m: np.ndarray) -> float:
                state = self._state_at(ControlVector(m, space), base=base)
                return self._linearize_state(state, gradient=False).value

            def gradient(m: np.ndarray) -> np.ndarray:
                state = self._state_at(ControlVector(m, space), base=base)
                result = self._linearize_state(state, gradient=True).gradient
                assert result is not None
                return result.values

            report["taylor"] = gradient_taylor_test(
                objective, gradient, lin.point.values, dv.values, steps=steps
            )
            passed = passed and bool(report["taylor"]["passed"])
        report["passed"] = passed
        return report


def _install_material(simulation: Any, block_id: str, values: np.ndarray) -> None:
    from frequensolve.model.parameterization import ParameterizedProperty

    for subdomain in simulation.model.subdomains:
        for name, prop in list(subdomain.properties.items()):
            if isinstance(prop, ParameterizedProperty) and prop.id == block_id:
                subdomain.properties[name] = prop.with_coefficients(values)
                return
    raise KeyError(f"no parameterized property carries block id {block_id!r}")


def _install_interface(simulation: Any, block_id: str, values: np.ndarray) -> None:
    surfaces = simulation.model.implicit_surfaces
    for index, surface in enumerate(list(surfaces)):
        control = getattr(surface, "control", None)
        if control is not None and getattr(control, "id", None) == block_id:
            surfaces[index] = surface.with_coefficients(values)
            return
    raise KeyError(f"no implicit surface carries control id {block_id!r}")


# ---------------------------------------------------------------------------
# linearization
# ---------------------------------------------------------------------------


class Linearization:
    """One saved ``fwi_operator`` state (one task per frequency).

    Attributes:
        problem: The view that produced it.
        space: Control space with the support mask applied.
        point: The linearization point on ``space``.
        state: The full baseline the state was saved at.
        frequencies: Frequencies of the saved tasks.
        fingerprint: Cache key.
        state_fingerprint: Sauce state fingerprint every operator input carries.
        registry_fingerprint: Sauce control registry fingerprint.
        entry: Cache bookkeeping (paths of the saved state).
        job: The ``linearize`` job.
        manifest: The ``fs-control-registry-1`` manifest.
        reports: Per-task objective reports.
        frequency_weights: One objective weight per task (unit by default).
        value: Total misfit value (weighted sum over tasks).
        report: ``term id -> weighted value`` summed over tasks.
        gradient: Real covector on ``space`` (smoothed when the problem has a
            smoothing), or ``None`` for a value-only linearization.
    """

    def __init__(
        self,
        problem: ImagingProblem,
        *,
        space: ControlSpace,
        state: ControlState,
        entry: LinearizationEntry,
        manifest: ControlRegistryManifest,
        reports: Sequence[ObjectiveReport],
        gradient_file: Optional[ControlVectorFile],
        support_masks: Mapping[str, np.ndarray],
    ) -> None:
        self.problem = problem
        self.space = space
        self.state = state
        self.point = state.vector(space)
        self.entry = entry
        self.job: FWIOperatorJob = entry.job
        self.frequencies: List[Any] = list(self.job.f_list)
        self.fingerprint = entry.fingerprint
        assert entry.state_fingerprint is not None
        assert entry.control_registry_fingerprint is not None
        self.state_fingerprint: str = entry.state_fingerprint
        self.registry_fingerprint: str = entry.control_registry_fingerprint
        self.manifest = manifest
        self.reports: List[ObjectiveReport] = list(reports)
        self.frequency_weights: np.ndarray = frequency_weights(
            self.job, problem.weights
        )
        self.value: float = total_value(self.reports, self.frequency_weights.tolist())
        merged: Dict[str, float] = {}
        for weight, report in zip(self.frequency_weights, self.reports):
            for term, weighted in report.weighted_values.items():
                merged[term] = merged.get(term, 0.0) + float(weight) * weighted
        self.report: Dict[str, float] = merged
        self.gradient: Optional[ControlVector] = (
            None if gradient_file is None else _vector_from_file(gradient_file, space)
        )
        self.support_masks: Dict[str, np.ndarray] = {
            name: np.array(mask, dtype=bool) for name, mask in support_masks.items()
        }
        self._data_space: Optional[DataSpace] = None
        self._jacobian: Optional[Jacobian] = None
        self._normal: Optional[Normal] = None
        self._ops: Dict[Tuple[str, str], Any] = {}
        self._counter = itertools.count(1)

    def __repr__(self) -> str:
        return (
            f"Linearization(value={self.value:.6g}, blocks={list(self.space.blocks)}, "
            f"frequencies={self.frequencies}, gradient={self.gradient is not None})"
        )

    # -- descriptors ----------------------------------------------------------

    @property
    def support(self) -> SupportMask:
        """Return the per-DOF support mask of ``space``."""

        return self.space.support

    @property
    def data_space(self) -> DataSpace:
        if self._data_space is None:
            self._data_space = DataSpace.from_simulation(
                self.problem.simulation, frequencies=self.frequencies
            )
        return self._data_space

    @property
    def jacobian(self) -> Jacobian:
        if self._jacobian is None:
            self._jacobian = Jacobian(self)
        return self._jacobian

    @property
    def normal(self) -> Normal:
        if self._normal is None:
            self._normal = Normal(self)
        return self._normal

    # -- operator actions -----------------------------------------------------

    def _control_vector(self, dv: Any) -> ControlVector:
        if isinstance(dv, ControlVector):
            if dv.space is not self.space and not dv.space.equivalent(self.space):
                raise ValueError("direction belongs to a different control space")
            return dv
        return ControlVector(dv, self.space)

    def _data_vector(self, r: Any) -> DataVector:
        if isinstance(r, DataVector):
            if r.space != self.data_space:
                raise ValueError("objective vector belongs to a different data space")
            return r
        return DataVector(np.asarray(r, dtype=self.data_space.dtype), self.data_space)

    def _ops_dir(self) -> Path:
        assert self.entry.directory is not None
        path = Path(self.entry.directory) / "ops" / f"{next(self._counter):04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _direction_file(self, dv: ControlVector, directory: Path) -> Path:
        return dv.to_file(
            state_fingerprint=self.state_fingerprint,
            registry_fingerprint=self.registry_fingerprint,
        ).write(directory / "direction.h5")

    def _memo(self, action: str, digest: str, compute: Callable[[], Any]) -> Any:
        key = (action, digest)
        if key not in self._ops:
            self._ops[key] = compute()
        return self._ops[key]

    def _family(
        self, action: str, per_task: Callable[[int], Dict[str, Any]]
    ) -> List[FWIOperatorJob]:
        """Build one single-frequency ``action`` job per saved task.

        Sauce reads ``fwi_operator.state`` of a derivative action exactly as
        given (only ``linearize`` appends the task suffix), so task ``t`` of
        the saved state is applied by its own job over ``frequencies[t-1]``.
        """

        return [
            self.problem._operator_job(
                self.space,
                action,
                frequencies=[frequency],
                state=self.job.state_file(task),
                **per_task(task),
            )
            for task, frequency in enumerate(self.frequencies, start=1)
        ]

    def _reduce(
        self, jobs: Sequence[FWIOperatorJob], *, weighted: bool
    ) -> ControlVector:
        total = self.space.zeros()
        for task, job in enumerate(jobs, start=1):
            part = _vector_from_file(reduce_covectors(job), self.space)
            if weighted:
                part = part * float(self.frequency_weights[task - 1])
            total = total + part
        return total

    def weight_data(self, r: Any) -> DataVector:
        """Return ``W r``: the data vector scaled by the per-frequency weights.

        ``W`` is the objective-side frequency weighting this linearization
        carries (unit unless the view set ``weights``); ``J.H @ W (J dv)``
        equals ``normal @ dv``.
        """

        dual = self._data_vector(r)
        values = np.array(dual.values, copy=True)
        for weight, frequency in zip(self.frequency_weights, self.frequencies):
            if weight == 1.0:
                continue
            for layout in self.data_space.term_layouts(frequency=frequency):
                values[layout.indices] *= float(weight)
        return DataVector(values, self.data_space)

    def jvp(self, dv: Any) -> DataVector:
        """Return ``J @ dv`` (memoized per direction)."""

        direction = self._control_vector(dv)

        def compute() -> DataVector:
            directory = self._ops_dir()
            path = self._direction_file(direction, directory)
            jobs = self._family(
                "jvp", lambda task: {"direction": path, "objective_vector": "jvp.json"}
            )
            self.problem._run_jobs(jobs)
            values = np.zeros(self.data_space.size, dtype=self.data_space.dtype)
            for job in jobs:
                values += read_task_objective_vectors(
                    job, self.data_space, state_fingerprint=self.state_fingerprint
                ).values
            return DataVector(values, self.data_space)

        return self._memo("jvp", _digest(direction.values), compute)

    def vjp(self, r: Any) -> ControlVector:
        """Return the real covector ``Re(J^H r)`` on ``space`` (memoized)."""

        dual = self._data_vector(r)

        def compute() -> ControlVector:
            directory = self._ops_dir()
            jobs = self._family(
                "vjp",
                lambda task: {
                    "objective_vector": directory / f"task_{task}" / "dual.json",
                    "covector": "vjp.h5",
                },
            )
            for job in jobs:
                # Inputs are read exactly as configured (no task suffix); each
                # single-frequency job receives the rows of its own frequency.
                assert job.objective_vector is not None
                dual.write_objective_vector(
                    job.objective_vector,
                    state_fingerprint=self.state_fingerprint,
                    term_layout=self.data_space.term_layouts(frequency=job.f_list[0]),
                    n_ranks=1,
                )
            self.problem._run_jobs(jobs)
            return self._reduce(jobs, weighted=False)

        return self._memo("vjp", _digest(dual.values), compute)

    def apply_normal(self, dv: Any) -> ControlVector:
        """Return ``Re(J^H W J) dv`` from Sauce's ``normal`` action (memoized)."""

        direction = self._control_vector(dv)

        def compute() -> ControlVector:
            directory = self._ops_dir()
            path = self._direction_file(direction, directory)
            jobs = self._family(
                "normal", lambda task: {"direction": path, "covector": "normal.h5"}
            )
            self.problem._run_jobs(jobs)
            return self._reduce(jobs, weighted=True)

        return self._memo("normal", _digest(direction.values), compute)
