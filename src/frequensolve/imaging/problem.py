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
    task_inputs,
    total_value,
)
from frequensolve.imaging.controls import (
    BoundControlSpace,
    ControlSpace,
    ControlState,
    ControlVector,
    ResolvedBlock,
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

# Block kinds whose authored baseline FrequenSolve installs in the simulation;
# the baseline of every other kind is Sauce's (``controls.state_output``).
_FRS_AUTHORED_KINDS = frozenset({"profile", "grid", "interface"})


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
        parent: Optional["_Shared"] = None,
        working_name: Optional[str] = None,
    ) -> None:
        self.name = str(name).strip()
        if not self.name or "/" in self.name:
            raise ValueError("ImagingProblem name must be a non-empty path segment")
        if not isinstance(controls, ControlSpace):
            controls = ControlSpace(controls)
        self.project = _project_path(simulation)
        # The caller's simulation (never mutated); ``with_controls`` rebinds a
        # new control layout against it.
        self.source_simulation = simulation
        # Jobs persist their simulation under ``<project>/simulations/<name>``;
        # work on a renamed copy so the authored simulation file stays untouched.
        working = simulation.copy(working_name or f"{simulation.name}__{self.name}")
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
        # Problems derived with ``with_controls`` share one backend (job name
        # counter), one LRU cache and the family list that lets an eviction
        # by any member forget the evicted linearization everywhere.
        self.family: List["_Shared"]
        self.cache: LinearizationCache
        if parent is not None:
            self._backend = parent.backend
            self.cache = parent.cache
            self.family = parent.family
        else:
            self.cache = LinearizationCache(self.workdir, cache_capacity)
            self.family = []
        self.family.append(self)
        self.linearizations: Dict[str, "Linearization"] = {}
        self.pending_manifest = not self.space.resolved
        # ``authored`` is the reference point Sauce runs when no
        # ``controls.state`` is staged.  Material and interface blocks are
        # authored by FrequenSolve (the bind installs their coefficients);
        # every other block (sources, reflectivity, mesh, registry) has a
        # baseline only Sauce knows, so until registry discovery those blocks
        # hold provisional placeholders (``ControlState.from_simulation``) and
        # ``sauce_owned`` makes the public state accessors discover first.
        self.authored: Optional[ControlState] = (
            None if self.pending_manifest else ControlState.from_simulation(self.space)
        )
        self.sauce_owned = self.pending_manifest or any(
            block.kind not in _FRS_AUTHORED_KINDS
            for block in self.space.resolved_blocks
        )
        self.state: Optional[ControlState] = self.authored
        # ``state`` still is the provisional ``authored`` (replaced on adoption).
        self.state_provisional = True
        # Sauce's ``controls.state`` must carry every registry block (source
        # blocks included); the complete baseline is learned from the first
        # ``state_output`` and merged with the space's blocks afterwards.
        self.baseline: Optional[ControlStateFile] = None
        self.manifest: Optional[ControlRegistryManifest] = None
        # Digest of the state whose coefficients are authored in the working
        # simulation (material/interface-only spaces, see ``_sync_simulation``).
        self.installed: Optional[str] = None
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
        self.state = ControlState(
            self.space.without_support(),
            state.values,
            scaling=state.scaling,
            scaling_units=state.scaling_units,
        )
        self.state_provisional = False

    def adopt_baseline(
        self, baseline: ControlStateFile, manifest: ControlRegistryManifest
    ) -> None:
        """Adopt Sauce's registry baseline as the authored reference point.

        Sauce-owned blocks (every kind but material profiles/lattices and
        interfaces) take their values, ``/scaling`` and ``/scaling_units``
        from ``baseline``; material and interface blocks keep FrequenSolve's
        authored values.  A still-provisional :attr:`state` becomes the new
        authored state; a state the caller set explicitly is kept.
        """

        self.manifest = manifest
        self.baseline = baseline
        if self.pending_manifest:
            self.space = self.space.with_manifest(manifest)  # type: ignore[assignment]
            self.pending_manifest = False
        full = self.space.without_support()
        base = (
            ControlState.from_simulation(full)
            if self.authored is None
            else self.authored
        )
        values = np.array(base.values, copy=True)
        scaling: Dict[str, float] = {}
        units: Dict[str, str] = {}
        for block, sl in zip(full.resolved_blocks, full.full_slices.values()):
            if block.kind in _FRS_AUTHORED_KINDS:
                continue
            if block.name not in baseline.blocks:
                raise ValueError(
                    f"Sauce's registry baseline has no block {block.name!r}"
                )
            exported = np.asarray(baseline[block.name], dtype=np.float64).reshape(-1)
            if exported.size != sl.stop - sl.start:
                raise ValueError(
                    f"block {block.name!r} has {sl.stop - sl.start} DOFs locally but "
                    f"{exported.size} in Sauce's registry baseline"
                )
            values[sl] = exported
            if block.name in baseline.scaling:
                scaling[block.name] = baseline.scaling[block.name]
                if block.name in baseline.scaling_units:
                    units[block.name] = baseline.scaling_units[block.name]
        self.authored = ControlState(full, values, scaling=scaling, scaling_units=units)
        if self.state_provisional:
            self.state = self.authored

    def forget(self, entries: Iterable[LinearizationEntry]) -> None:
        for entry in entries:
            for member in self.family:
                member.linearizations.pop(entry.fingerprint, None)


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
        self._init_view()

    def _init_view(self) -> None:
        """Initialize the view-level attributes of a full (unrestricted) problem."""

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
        """Return the current full baseline (shared by every view).

        Source, reflectivity, mesh and registry blocks have baselines only
        Sauce knows (``controls.state_output``).  On a space with such blocks
        the first access runs registry discovery (one value-only linearize at
        the authored point, see :meth:`linearize`) so the state never exposes
        FrequenSolve's provisional placeholders; material/interface-only
        spaces are known locally and submit nothing.
        """

        self._ensure_registry()
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

    def with_controls(
        self, controls: Any, *, state: Optional[ControlState] = None
    ) -> "ImagingProblem":
        """Return a problem over another control layout (resolution change).

        The new problem shares this problem's caller simulation, site,
        backend (job naming), observed data, misfit, frequencies, smoothing,
        support threshold, workdir root and linearization cache; its
        linearizations are keyed by an identity that includes the new
        control layout, so they never collide with this problem's.  It is a
        full problem: this view's active blocks, frequency subset and
        overrides are not carried over (stage views re-apply them).

        Args:
            controls: A complete :class:`ControlSpace` (or single block), or a
                mapping ``{block key: new block spec}`` replacing those blocks
                of :attr:`full_space` and keeping the others (e.g.
                ``{"vp": im.DepthProfile("vp", "sediment", spacing=25*u.m)}``).
            state: The initial state.  ``None`` transfers the current state;
                a state on this problem's layout is transferred; a state on
                the new layout is adopted as is.

        The transfer is block-wise: blocks whose layout is unchanged are
        copied, material profile / lattice blocks are projected with
        :meth:`ControlSpace.transfer_to` (exact for fields the new basis
        represents, e.g. a hat profile refined by node bisection), blocks
        absent from this problem take the new problem's authored values, and
        interface, source, reflectivity, mesh and registry blocks must keep
        their layout (``ValueError`` otherwise).

        Raises:
            KeyError: A mapping names a block key this problem does not have.
            ValueError: The layouts cannot be transferred, or the new space
                has mesh blocks whose layout Sauce has not reported yet.
        """

        shared = self._shared
        old_full = shared.space
        if isinstance(controls, Mapping) and not isinstance(controls, ControlSpace):
            unknown = [key for key in controls if key not in old_full.keys]
            if unknown:
                raise KeyError(
                    f"with_controls names unknown block keys {unknown}; the "
                    f"problem has {list(old_full.keys)}"
                )
            specs = old_full.specs
            space = ControlSpace(
                **{key: controls.get(key, spec) for key, spec in specs.items()}
            )
        elif isinstance(controls, ControlSpace):
            space = controls
        else:
            space = ControlSpace(controls)
        layout = fingerprint(
            space=[[key, repr(spec)] for key, spec in space.specs.items()]
        )
        source = shared.source_simulation
        derived = _Shared(
            source,
            controls=space,
            observed=shared.observed_source,
            misfit=shared.misfit,
            frequencies=shared.frequencies,
            site=shared._site,
            smoothing=shared.smoothing,
            workdir=shared.workdir,
            name=shared.name,
            min_support=shared.min_support,
            submit_options=shared.submit_options,
            cache_capacity=shared.cache.capacity,
            parent=shared,
            working_name=f"{source.name}__{shared.name}__{layout.split(':')[-1][:10]}",
        )
        if derived.pending_manifest:
            derived.family.remove(derived)
            raise ValueError(
                "with_controls cannot build a layout with mesh blocks before Sauce "
                "reports it; keep mesh blocks out of resolution changes"
            )
        problem = object.__new__(ImagingProblem)
        problem._shared = derived
        problem._init_view()
        new_full = derived.space.without_support()
        if state is None:
            state = self._require_state()
        if state.space.blocks == new_full.blocks and state.size == new_full.full_size:
            derived.set_state(state)
        else:
            derived.set_state(_transfer_state(state, old_full, derived))
        return problem

    # -- states and vectors ---------------------------------------------------

    def _ensure_registry(self) -> None:
        """Run registry discovery if Sauce owns baselines this problem lacks."""

        shared = self._shared
        if shared.baseline is None and shared.sauce_owned:
            self._discover_registry()

    def _require_state(self, *, discover: bool = True) -> ControlState:
        if discover:
            self._ensure_registry()
        state = self._shared.state
        if state is None:
            raise ValueError(
                "the control state is unknown until the first linearize supplies "
                "the registry manifest (mesh controls)"
            )
        return state

    def _state_at(
        self,
        v: Any,
        base: Optional[ControlState] = None,
        *,
        discover: bool = True,
    ) -> ControlState:
        """Return the full state at ``v`` (vector on this view's space).

        ``discover=False`` skips registry discovery (``dry_run`` submits
        nothing), leaving Sauce-owned blocks at their provisional values.
        """

        if isinstance(v, ControlState):
            if v.space.blocks != self._shared.space.blocks:
                raise ValueError("state does not cover the problem's control blocks")
            return v
        state = self._require_state(discover=discover) if base is None else base
        if v is None:
            return state
        vector = v if isinstance(v, ControlVector) else ControlVector(v, self.space)
        return state.with_update(vector)

    def _linearization_point(self, v: Any) -> Optional[ControlState]:
        """Return the state a linearize at ``v`` runs (``None``: pending layout).

        ``v=None`` before registry discovery is the authored point, which
        discovers the registry in its own job; anything else goes through
        :meth:`_state_at` (discovering first when Sauce owns baselines).
        """

        shared = self._shared
        if v is None and (
            shared.state is None
            or (shared.baseline is None and shared.state_provisional)
        ):
            return shared.state
        return self._state_at(v)

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
        discover: bool = False,
    ) -> FWIOperatorJob:
        shared = self._shared
        smoothing = self.smoothing if gradient else None
        # Only the registry discovery linearize asks for ``state_output`` and
        # ``manifest`` (task-suffixed when the job has several tasks).
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
            state_output="baseline.h5" if discover else None,
            manifest="registry.json" if discover else None,
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

    def _inline_authoring(self) -> bool:
        """Return whether points are authored into the simulation, not a state file.

        Sauce's ``controls.state`` must carry every registry block.  A space
        made only of material and interface blocks avoids the file: the
        coefficients are installed in the working simulation (Sauce's
        authored baseline is then the point itself).  Other spaces stage
        ``controls.state`` built on the discovered registry baseline; its
        mechanism ``/scaling`` lets every frequency task replay it.
        """

        space = self._shared.space
        if not space.resolved:
            return False
        return all(block.kind in _FRS_AUTHORED_KINDS for block in space.resolved_blocks)

    def _sync_simulation(self, state: Optional[ControlState]) -> None:
        """Author ``state``'s coefficients in the working simulation when inline."""

        shared = self._shared
        if state is None or not self._inline_authoring():
            return
        digest = _digest(state.values)
        if shared.installed == digest:
            return
        simulation = shared.simulation
        full = shared.space
        for block, sl in zip(full.resolved_blocks, full.full_slices.values()):
            values = np.array(state.values[sl], copy=True)
            block_id = unqualified_block_name(block.name)
            if block.kind == "interface":
                _install_interface(simulation, block_id, values)
            else:
                _install_material(simulation, block_id, values)
        simulation.save()
        shared.installed = digest

    def _stage_state(
        self, key: str, state: Optional[ControlState]
    ) -> Tuple[Path, Optional[Path]]:
        """Stage ``controls.state`` for ``state``; ``None`` runs the authored baseline.

        Sauce requires the complete registry baseline, so a state can only be
        written once :attr:`_Shared.baseline` is known (after the first
        linearize).  The authored baseline itself needs no file, and neither
        does a point authored inline (:meth:`_inline_authoring`).
        """

        stage = self.workdir / key.split(":")[-1][:16]
        stage.mkdir(parents=True, exist_ok=True)
        baseline = self._shared.baseline
        if (
            state is None
            or self._is_authored(state)
            or baseline is None
            or self._inline_authoring()
        ):
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
        # ``/scaling`` states the coordinate scale of each mechanism block so
        # Sauce can replay it in any frequency task: blocks taken from the
        # registry baseline keep its scaling, blocks of ``state`` use the
        # state's (a block without scaling is read in task coordinates).
        own = set(state.space.blocks)
        scaling = {
            name: value for name, value in baseline.scaling.items() if name not in own
        }
        units = {
            name: value
            for name, value in baseline.scaling_units.items()
            if name in scaling
        }
        scaling.update(state.scaling)
        units.update(state.scaling_units)
        path = stage / "state.h5"
        ControlStateFile(blocks, scaling=scaling, scaling_units=units).write(path)
        return stage, path

    def _read_masks(
        self, job: FWIOperatorJob, space: ControlSpace
    ) -> Optional[Dict[str, np.ndarray]]:
        """Return Sauce's support masks for ``space`` or ``None`` when absent."""

        file: Any = None
        if job.state_output is not None and job.state_output_file(1).is_file():
            file = read_state_output(job)
        elif job.covector is not None and job.covector_file(1).is_file():
            file = ControlVectorFile.read(job.covector_file(1), native=False)
        elif self._shared.baseline is not None:
            file = self._shared.baseline
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
        discover = shared.baseline is None
        if discover and not self._is_authored(state):
            # A point away from the authored one needs the complete registry
            # baseline for its ``controls.state``: discover it first with a
            # value-only linearize at the authored point.
            self._discover_registry()
            discover = False
            if state is None:
                state = shared.state
                key = self._fingerprint(state)
                cached = shared.linearizations.get(key)
                if cached is not None and (cached.gradient is not None or not gradient):
                    shared.cache.get(key)
                    return cached
        space = self.space
        self._sync_simulation(state)
        stage, control_state = self._stage_state(key, state)
        job = self._linearize_job(
            space, control_state, gradient=gradient, discover=discover
        )
        self._run_job(job)

        if discover:
            # The job ran without ``controls.state``, i.e. at Sauce's own
            # baseline: adopt it (every task exports the complete baseline;
            # mechanism blocks carry ``/scaling`` so task 1's replays in any
            # task) and key the linearization by the adopted point.
            shared.adopt_baseline(read_state_output(job), read_manifest(job))
            state = shared.authored
            key = self._fingerprint(state)
            space = self.space
        assert shared.manifest is not None
        manifest = shared.manifest
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

    def _discover_registry(self) -> None:
        """Learn Sauce's complete registry baseline (``state_output``/``manifest``).

        Runs one value-only linearize of this view at the authored point
        (one job over the view's frequencies; each task exports its baseline
        and manifest under ``<stem>_<task><ext>``) and adopts task 1's
        baseline (:meth:`_Shared.adopt_baseline`).  The linearization is
        cached like any other, so a following value-only linearize at the
        authored point reuses it.
        """

        shared = self._shared
        self._linearize_state(shared.authored, gradient=False)
        assert shared.baseline is not None

    def linearize(self, v: Any = None, *, gradient: bool = True) -> "Linearization":
        """Save (or reuse) the Sauce state at ``v`` and return its linearization.

        Args:
            v: Point on this view's space (``ControlVector`` or array), a full
                ``ControlState``, or ``None`` for the current state.
            gradient: Request the covector.  A cached gradient-carrying
                linearization of the same point is reused either way.

        The first linearize of a problem learns Sauce's complete registry
        baseline from ``state_output``/``manifest``: a linearize at the
        authored point does so in its own job; a non-authored point first runs
        one value-only linearize at the authored point.  Source, reflectivity,
        mesh and registry blocks of :attr:`state` then carry Sauce's baseline
        (with the mechanism ``/scaling``); material and interface blocks keep
        FrequenSolve's authored values.  Asking for :attr:`state`,
        :meth:`vector`, :meth:`state_from` or a point ``v`` on a space with
        Sauce-owned blocks runs that discovery first, so no caller sees (or
        steps from) the provisional placeholders.  A point of a
        material/interface-only space is authored into the working
        simulation; other spaces stage it as ``controls.state``.  Each action
        is one job over the view's frequencies (one task per frequency).
        """

        return self._linearize_state(self._linearization_point(v), gradient=gradient)

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

        Material (profile, lattice) and interface blocks are always installed.
        Every other block of the full space is compared with its authored
        baseline and skipped when unchanged.  Changed source blocks:

        - ``source.<i>.position`` moves the (inline) physical source point.
        - ``source.<i>.mechanism`` is installed when its coordinate scale is
          known (``/scaling/<block>`` of the state or of Sauce's registry
          baseline): the physical components ``coordinate * scaling`` in
          ``/scaling_units/<block>`` become the inline source's basis --
          ``amplitude`` for scalar/monopole kinds, unit ``direction`` plus
          ``amplitude`` for vector/dipole kinds, and a
          ``moment_tensor`` mechanism (``xx/zz/xz`` in 2D, embedded in the
          x-z plane; ``xx/yy/zz/yz/xz/xy`` in 3D) for tensor kinds.  Complex
          coordinates must share one phase; the phase is applied like a
          signature.  Without scaling the coordinates are task-dependent and
          :class:`NotImplementedError` is raised.
        - ``source.<i>.signature`` ``q`` multiplies source ``i``'s column of
          the source encoding (``C = E diag(q)``; an identity encoding is made
          explicit first).  ``q`` is frequency independent, so a complex ``q``
          applies a frequency-independent gain ``|q|`` and phase ``arg q`` to
          the source.
        - ``source.<i>.signature_df`` (an additive per-Hz term), reflectivity
          maps and mesh blocks have no representation in the authored
          simulation and raise :class:`NotImplementedError` naming the block.
        """

        state = self._state_at(v)
        simulation = copy.deepcopy(self.simulation)
        full = self._shared.space
        factors: Dict[int, complex] = {}
        for block, sl in zip(full.resolved_blocks, full.full_slices.values()):
            values = np.array(state.values[sl], copy=True)
            if block.kind in {"profile", "grid"}:
                _install_material(
                    simulation, unqualified_block_name(block.name), values
                )
                continue
            if block.kind == "interface":
                _install_interface(
                    simulation, unqualified_block_name(block.name), values
                )
                continue
            baseline = self._authored_block(block.name, sl)
            if baseline is not None and np.array_equal(values, baseline):
                continue
            if block.kind == "source" and block.quantity == "position":
                _install_source_position(simulation, block, values)
                continue
            source_id = int(block.source_id or 0)
            if block.kind == "source" and block.quantity == "signature":
                q = complex(values[0], values[1])
                factors[source_id] = factors.get(source_id, 1.0 + 0.0j) * q
                continue
            if block.kind == "source" and block.quantity == "mechanism":
                scale, units = self._mechanism_scaling(state, block.name)
                if scale is None:
                    raise NotImplementedError(_uninstallable_message(block))
                phase = _install_source_mechanism(
                    simulation, block, values, scale, units
                )
                if phase != 1.0:
                    factors[source_id] = factors.get(source_id, 1.0 + 0.0j) * phase
                continue
            raise NotImplementedError(_uninstallable_message(block))
        for source_id, factor in sorted(factors.items()):
            _scale_source(simulation, source_id, factor)
        return simulation

    def _mechanism_scaling(
        self, state: ControlState, name: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Return ``(scaling, units)`` of a mechanism block (state, then baseline)."""

        if name in state.scaling:
            return state.scaling[name], state.scaling_units.get(name)
        baseline = self._shared.baseline
        if baseline is not None and name in baseline.scaling:
            return baseline.scaling[name], baseline.scaling_units.get(name)
        return None, None

    def _authored_block(self, name: str, sl: slice) -> Optional[np.ndarray]:
        """Return the authored baseline of block ``name`` (Sauce layout), if known.

        The authored state (the bind's baselines) is the reference; before it
        is known (pending mesh sizes) Sauce's discovered registry baseline is.
        """

        authored = self._shared.authored
        if authored is not None:
            return np.asarray(authored.values[sl])
        baseline = self._shared.baseline
        if baseline is not None and name in baseline.names:
            return np.asarray(baseline[name], dtype=np.float64)
        return None

    # -- diagnostics ----------------------------------------------------------

    def dry_run(self, v: Any = None) -> Dict[str, Any]:
        """Describe the linearize job at ``v`` without submitting it.

        Nothing is submitted, not even registry discovery: before the first
        linearize the complete registry baseline is unknown, so a non-authored
        ``v`` is described without ``controls.state`` and
        ``payload["registry_discovery"]`` is ``True`` (a discovery linearize
        at the authored point would run first).
        """

        state = (
            None
            if (v is None and self._shared.state is None)
            else self._state_at(v, discover=False)
        )
        key = self._fingerprint(state)
        _stage, control_state = self._stage_state(key, state)
        job = self._linearize_job(
            self.space,
            control_state,
            gradient=True,
            discover=self._shared.baseline is None,
        )
        payload = self.backend.dry_run(job)
        payload["fingerprint"] = key
        payload["registry_discovery"] = bool(
            self._shared.baseline is None and not self._is_authored(state)
        )
        return payload

    def _reflectivity_rules(
        self, blocks: Sequence[Any], comparisons: set, extension: Any
    ) -> Tuple[List[str], List[str]]:
        """Return ``(errors, warnings)`` of a reflectivity-carrying space.

        ``fwi_operator.reflectivity`` needs full-dimensional first-order
        acoustic or classic elastic DPG with unrelaxed assembly and waveform
        objectives; it rejects ``extension``, coupled physics, Galerkin,
        2.5D, axisymmetry, phase objectives, interface (geometry) controls and
        ``signature_df`` source blocks (spectral observation support).
        """

        errors: List[str] = []
        warnings: List[str] = []
        simulation = self._shared.simulation
        if extension is not None:
            errors.append(
                "reflectivity and extension are mutually exclusive in one problem"
            )
        phase = sorted(comparisons - {"waveform"})
        if phase:
            errors.append(
                "reflectivity requires waveform comparisons " f"(misfit uses {phase})"
            )
        interface = [b.name for b in blocks if b.kind == "interface"]
        if interface:
            errors.append(
                "reflectivity rejects interface (geometry) controls: "
                + ", ".join(interface)
            )
        spectral = [
            b.name
            for b in blocks
            if b.kind == "source" and b.quantity == "signature_df"
        ]
        if spectral:
            errors.append(
                "reflectivity rejects signature_df source blocks: "
                + ", ".join(spectral)
            )
        physics = str(getattr(simulation, "physics", "") or "")
        if physics not in {"acoustic", "elastic"}:
            errors.append(
                "reflectivity requires acoustic or elastic physics "
                f"(simulation uses {physics!r})"
            )
        if getattr(simulation, "_axisymmetric", False):
            errors.append("reflectivity rejects axisymmetric simulations")
        dimension = getattr(simulation, "dimension", None)
        if dimension not in (2, 3):
            errors.append(
                f"reflectivity requires a 2D or 3D simulation (dimension {dimension!r})"
            )
        discretization = getattr(simulation, "discretization", None)
        method = str(
            (getattr(discretization, "extra", None) or {}).get("method", "DPG")
        )
        if method.strip().upper() != "DPG":
            errors.append(
                f"reflectivity requires Discretization(method='DPG') (got {method!r})"
            )
        solver = getattr(getattr(simulation, "solver", None), "extra", None) or {}
        relaxed = solver.get("relaxed_assembly")
        if relaxed is None:
            relaxed = str(solver.get("mode", "")).strip().lower() == "fast"
        if relaxed:
            errors.append(
                "reflectivity requires unrelaxed assembly "
                "(SolverConfig(relaxed_assembly=False))"
            )
        return errors, warnings

    def capabilities(self, *, extension: Any = None) -> Dict[str, Any]:
        """Statically validate the block/misfit/physics combination.

        Returns a mapping with ``errors`` (combinations Sauce rejects) and
        ``warnings`` (documented limitations); ``ok`` is ``not errors``.
        ``extension`` (an :class:`Extension`, when the caller is an extended
        problem) is checked against the reflectivity blocks, which Sauce
        rejects alongside ``fwi_operator.extension``.
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
        if "reflectivity" in kinds:
            rejected, limited = self._reflectivity_rules(blocks, comparisons, extension)
            errors.extend(rejected)
            warnings.extend(limited)
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

    def extend(self, extension: Any) -> Any:
        """Return an :class:`~frequensolve.imaging.extension.ExtendedProblem`.

        The extended problem shares this view's state, cache, backend and
        simulation and exposes Sauce's auxiliary model extension (FWIME)
        actions; see :mod:`frequensolve.imaging.extension`.
        """

        from frequensolve.imaging.extension import ExtendedProblem

        return ExtendedProblem(self, extension)

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


_FIXED_LAYOUT_KINDS = {"interface", "source", "reflectivity", "mesh", "registry"}


def _transfer_state(
    state: ControlState, old: ControlSpace, shared: "_Shared"
) -> ControlState:
    """Transfer a full state on ``old`` to ``shared.space`` block by block."""

    source_space = old.without_support()
    if (
        state.space.blocks != source_space.blocks
        or state.size != source_space.full_size
    ):
        raise ValueError(
            "with_controls state covers neither the current nor the new layout"
        )
    target_space = shared.space.without_support()
    authored = shared.authored
    by_key = {(b.key, b.address): b for b in source_space.resolved_blocks}
    by_name = {b.name: b for b in source_space.resolved_blocks}
    values = np.zeros(target_space.full_size, dtype=np.float64)
    transfers: List[str] = []
    for target, sl in zip(
        target_space.resolved_blocks, target_space.full_slices.values()
    ):
        source = by_key.get((target.key, target.address)) or by_name.get(target.name)
        if source is None:
            if authored is None:
                raise ValueError(
                    f"block {target.name!r} is new and has no authored baseline"
                )
            values[sl] = authored.values[sl]
            continue
        old_values = state.values[source_space.full_slices[source.name]]
        same = (
            source.size == target.size
            and source.complex == target.complex
            and source.kind == target.kind
            and _same_layout(source, target)
        )
        if same:
            values[sl] = old_values
            continue
        if target.kind in _FIXED_LAYOUT_KINDS or source.kind in _FIXED_LAYOUT_KINDS:
            raise ValueError(
                f"with_controls cannot change the layout of {target.kind} block "
                f"{target.name!r} ({source.size} -> {target.size} DOFs); only "
                "material profile and lattice blocks change resolution"
            )
        transfers.append(target.name)
    if transfers:
        # ``transfer_to`` projects every material block of a two-space pair;
        # restrict both to the blocks that actually change.
        mine = source_space.restrict(
            [
                (by_key.get((b.key, b.address)) or by_name[b.name]).name
                for b in target_space.resolved_blocks
                if b.name in transfers
            ]
        )
        theirs = target_space.restrict(transfers)
        moved = mine.transfer_to(theirs, state.vector(mine))
        full = theirs.to_sauce_vector(moved)
        for name, sl in theirs.full_slices.items():
            values[target_space.full_slices[name]] = full[sl]
    return ControlState(
        target_space, values, scaling=state.scaling, scaling_units=state.scaling_units
    )


def _same_layout(a: ResolvedBlock, b: ResolvedBlock) -> bool:
    if a.control is None or b.control is None:
        return a.control is None and b.control is None
    try:
        return bool(a.control.to_fs() == b.control.to_fs())
    except Exception:
        return False


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


def _install_source_position(
    simulation: Any, block: ResolvedBlock, values: np.ndarray
) -> None:
    """Move physical source ``block.source_id`` to ``values`` (authored frame)."""

    geometry = getattr(simulation.acquisition, "source_geometry", None)
    source_id = int(block.source_id or 0)
    if geometry is None or geometry.geometry_type != "Inline":
        kind = "none" if geometry is None else geometry.geometry_type
        raise NotImplementedError(
            f"source block {block.name!r} differs from its authored baseline, but "
            f"the acquisition's source geometry is {kind}; only inline source "
            "points can be moved by simulation_at"
        )
    try:
        geometry.set_point_coordinates(source_id - 1, values)
    except (IndexError, ValueError) as exc:
        raise NotImplementedError(
            f"source block {block.name!r}: cannot move physical source "
            f"{source_id}: {exc}"
        ) from exc


_MECHANISM_PHASE_TOLERANCE = 1.0e-9


def _inline_point(simulation: Any, block: ResolvedBlock) -> Tuple[Any, Any, int]:
    """Return ``(geometry, point, index)`` of an inline physical source."""

    geometry = getattr(simulation.acquisition, "source_geometry", None)
    if geometry is None or geometry.geometry_type != "Inline":
        kind = "none" if geometry is None else geometry.geometry_type
        raise NotImplementedError(
            f"source block {block.name!r} differs from its authored baseline, but "
            f"the acquisition's source geometry is {kind}; only inline sources "
            "can carry a per-source mechanism"
        )
    index = int(block.source_id or 0) - 1
    count = int(geometry.point_count or 0)
    if index < 0 or index >= count:
        raise NotImplementedError(
            f"source block {block.name!r}: the geometry has {count} sources"
        )
    return geometry, geometry.sources[index], index


def _real_components(values: np.ndarray, name: str) -> Tuple[np.ndarray, complex]:
    """Split complex coordinates ``c`` into real ``r`` and a phase with ``c = r * phase``.

    The phase is that of the largest component folded into the half-plane
    ``Re(phase) > 0`` (or ``phase = i``), so real coordinates keep their signs
    and get ``phase = 1``.  Components with different phases have no real
    representation.
    """

    coords = np.asarray(values[0::2], dtype=np.float64) + 1j * np.asarray(
        values[1::2], dtype=np.float64
    )
    magnitude = np.abs(coords)
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak == 0.0:
        return np.zeros(coords.size), 1.0 + 0.0j
    phase = complex(coords[int(np.argmax(magnitude))] / peak)
    if phase.real < 0.0 or (phase.real == 0.0 and phase.imag < 0.0):
        phase = -phase
    if abs(phase - 1.0) <= _MECHANISM_PHASE_TOLERANCE:
        phase = 1.0 + 0.0j
    real = coords / phase
    if float(np.max(np.abs(real.imag))) > _MECHANISM_PHASE_TOLERANCE * peak:
        raise NotImplementedError(
            f"source block {name!r} has complex mechanism components with "
            "different phases; only a real mechanism times one common phase can "
            "be authored (the phase is installed like a signature)"
        )
    return np.asarray(real.real, dtype=np.float64), phase


def _install_source_mechanism(
    simulation: Any,
    block: ResolvedBlock,
    values: np.ndarray,
    scale: float,
    units: Optional[str],
) -> complex:
    """Author ``coordinate * scale`` as source ``block.source_id``'s basis.

    Returns the common phase of the complex coordinates; the caller applies
    it to the source encoding like a signature.
    """

    geometry, point, _index = _inline_point(simulation, block)
    kind = str(point.kind or geometry.kind).strip().lower()
    dimension = int(simulation.dimension)
    real, phase = _real_components(values, block.name)
    physical = real * float(scale)
    if kind in {"scalar", "monopole"}:
        if physical.size != 1:
            raise ValueError(f"{block.name!r} must have one component for {kind}")
        point.amplitude = {"value": float(physical[0]), "units": units or "N*m"}
        return phase
    if kind in {"vector", "dipole"}:
        if physical.size != dimension:
            raise ValueError(
                f"{block.name!r} must have {dimension} components for {kind}"
            )
        strength = float(np.linalg.norm(physical))
        default_units = "N" if kind == "vector" else "N*m"
        if strength > 0.0:
            point.direction = (physical / strength).tolist()
        point.amplitude = {"value": strength, "units": units or default_units}
        return phase
    if kind == "tensor":
        tensor = np.zeros((3, 3), dtype=np.float64)
        if dimension == 2 and physical.size == 3:
            xx, zz, xz = physical
            tensor[0, 0], tensor[2, 2] = xx, zz
            tensor[0, 2] = tensor[2, 0] = xz
        elif dimension == 3 and physical.size == 6:
            xx, yy, zz, yz, xz, xy = physical
            tensor[0, 0], tensor[1, 1], tensor[2, 2] = xx, yy, zz
            tensor[1, 2] = tensor[2, 1] = yz
            tensor[0, 2] = tensor[2, 0] = xz
            tensor[0, 1] = tensor[1, 0] = xy
        else:
            raise ValueError(
                f"{block.name!r} has {physical.size} components; a {dimension}-D "
                "tensor source needs " + ("3" if dimension == 2 else "6")
            )
        point.mechanism = {
            "type": "moment_tensor",
            "tensor": tensor.tolist(),
            "units": units or "N*m",
        }
        # The raw tensor entries are the physical moment components; a
        # separate strength would renormalize them.
        point.amplitude = None
        point.extra.pop("moment_magnitude", None)
        return phase
    raise NotImplementedError(
        f"source block {block.name!r}: cannot author a mechanism for {kind!r} sources"
    )


def _scale_source(simulation: Any, source_id: int, factor: complex) -> None:
    """Multiply physical source ``source_id`` by ``factor`` through the encoding.

    ``C = E diag(q)``: the source's column of the source encoding is scaled.
    Without an encoding (identity, one field per source named after it) the
    identity is made explicit as a dense encoding first.
    """

    if factor == 1.0:
        return
    from frequensolve.seismic.sources import SourceEncoding

    acquisition = simulation.acquisition
    geometry = getattr(acquisition, "source_geometry", None)
    if geometry is None:
        raise NotImplementedError("the acquisition has no physical sources to scale")
    index = int(source_id) - 1
    names = acquisition.source_point_names()
    encoding = acquisition.source_encoding
    if encoding is None:
        count = acquisition.known_source_point_count()
        if count is None or len(names) != int(count):
            raise NotImplementedError(
                f"cannot make the identity encoding of source {source_id} explicit: "
                "the physical source catalog is not known locally"
            )
        encoding = SourceEncoding.dense(
            np.eye(int(count), dtype=np.complex128), names=names
        )
    try:
        scaled = encoding.scaled_source(
            index,
            factor,
            source_name=names[index] if 0 <= index < len(names) else None,
        )
    except (ValueError, IndexError) as exc:
        raise NotImplementedError(
            f"cannot apply the signature of source {source_id}: {exc}"
        ) from exc
    acquisition.source_encoding = scaled


def _uninstallable_message(block: ResolvedBlock) -> str:
    """Return why a changed non-material block cannot be authored."""

    if block.kind == "source" and block.quantity == "mechanism":
        return (
            f"source block {block.name!r} differs from its authored baseline, but "
            "no /scaling is known for it: Sauce's mechanism coordinates are in "
            "the writing task's nondimensional units, and only a state export "
            "with /scaling/<block> (Sauce >= imaging-api/multitask-operators) "
            "converts them to physical source strengths"
        )
    if block.kind == "source":
        return (
            f"source block {block.name!r} differs from its authored baseline; "
            f"the {block.quantity} coefficients have no representation on the "
            "authored sources (signature_df is an additive per-Hz term and "
            "FrequenSolve authors no Acquisition/source_signature spectrum), so "
            "simulation_at cannot install them"
        )
    if block.kind == "mesh":
        return (
            f"mesh block {block.name!r} differs from its authored baseline; "
            "installing mesh control coefficients requires a control checkpoint"
        )
    if block.kind == "reflectivity":
        return (
            f"reflectivity block {block.name!r} differs from its authored "
            "baseline; reflectivity maps have no representation in the authored "
            "simulation"
        )
    return (
        f"{block.kind} block {block.name!r} differs from its authored baseline "
        "and cannot be installed in the simulation"
    )


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
        # Sauce fingerprints every frequency task's saved objective state on
        # its own (per-frequency mesh adaptation makes them differ) and checks
        # each task's objective-vector input against its own fingerprint, so
        # operator inputs are written per task.
        self.task_fingerprints: List[Tuple[str, str]] = self._task_fingerprints()
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

    def _with_covector(self) -> "Linearization":
        """Return a linearization of this point that carries covector parts.

        Sauce checks every direction against its task's control registry
        fingerprint, and only the covector parts of a ``linearize`` reveal
        the per-task fingerprints (mesh adaptation makes them differ per
        frequency), so operators hang off a gradient-carrying linearization.
        """

        if self.gradient is not None:
            return self
        return self.problem._linearize_state(self.state, gradient=True)

    @property
    def jacobian(self) -> Jacobian:
        if self._jacobian is None:
            self._jacobian = Jacobian(self._with_covector())
        return self._jacobian

    @property
    def normal(self) -> Normal:
        if self._normal is None:
            self._normal = Normal(self._with_covector())
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

    def _task_fingerprints(self) -> List[Tuple[str, str]]:
        """Return ``(state, registry)`` Sauce fingerprints of every saved task."""

        job = self.job
        pairs: List[Tuple[str, str]] = []
        for task, report in enumerate(self.reports, start=1):
            state_fp: Optional[str] = None
            registry_fp: Optional[str] = None
            if job.covector is not None and job.covector_file(task).is_file():
                part = ControlVectorFile.read(job.covector_file(task), native=False)
                state_fp = part.state_fingerprint
                registry_fp = part.control_registry_fingerprint
            if state_fp is None:
                state_fp = report.state_fingerprint
            pairs.append(
                (
                    state_fp or self.state_fingerprint,
                    registry_fp or self.registry_fingerprint,
                )
            )
        return pairs

    def _direction_stem(self, dv: ControlVector, directory: Path) -> Path:
        """Write ``dv`` once per task under its fingerprints; return the job input.

        Sauce resolves a direction input of a multi-task job to the
        task-suffixed sibling ``<stem>_<task><ext>``, so task ``t`` reads its
        own copy (each carries that task's state/registry fingerprints).
        """

        stem = directory / "direction.h5"
        for task, path in task_inputs(stem, len(self.frequencies)):
            state_fp, registry_fp = self.task_fingerprints[task - 1]
            dv.to_file(
                state_fingerprint=state_fp, registry_fingerprint=registry_fp
            ).write(path)
        return stem

    def _objective_vector_stem(self, dual: DataVector, directory: Path) -> Path:
        """Write each task's rows of ``dual`` under its fingerprint; return the input."""

        stem = directory / "dual.json"
        for task, path in task_inputs(stem, len(self.frequencies)):
            dual.write_objective_vector(
                path,
                state_fingerprint=self.task_fingerprints[task - 1][0],
                term_layout=self.data_space.term_layouts(
                    frequency=self.frequencies[task - 1]
                ),
                n_ranks=1,
            )
        return stem

    def _memo(self, action: str, digest: str, compute: Callable[[], Any]) -> Any:
        key = (action, digest)
        if key not in self._ops:
            self._ops[key] = compute()
        return self._ops[key]

    def _state_input(self) -> Path:
        """Return the saved-state input of a derivative action on this point.

        ``linearize`` always writes ``<state stem>_<task><ext>``; a multi-task
        derivative job names the stem (Sauce resolves each task's sibling)
        and a single-task job names task 1's file exactly.
        """

        return (
            self.job.state_file(1)
            if len(self.frequencies) == 1
            else self.job.state_file()
        )

    def _action_job(self, action: str, **options: Any) -> FWIOperatorJob:
        """Build the one ``action`` job over every saved task of this point.

        The working simulation is re-synchronized to this linearization's
        point first (a later linearize may have moved it).
        """

        self.problem._sync_simulation(self.state)
        return self.problem._operator_job(
            self.space,
            action,
            frequencies=self.frequencies,
            state=self._state_input(),
            **options,
        )

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
            job = self._action_job(
                "jvp",
                direction=self._direction_stem(direction, directory),
                objective_vector="jvp.json",
            )
            self.problem._run_job(job)
            return read_task_objective_vectors(
                job,
                self.data_space,
                state_fingerprint=[fp for fp, _ in self.task_fingerprints],
            )

        return self._memo("jvp", _digest(direction.values), compute)

    def vjp(self, r: Any) -> ControlVector:
        """Return the real covector ``Re(J^H r)`` on ``space`` (memoized)."""

        dual = self._data_vector(r)

        def compute() -> ControlVector:
            directory = self._ops_dir()
            job = self._action_job(
                "vjp",
                objective_vector=self._objective_vector_stem(dual, directory),
                covector="vjp.h5",
            )
            self.problem._run_job(job)
            return _vector_from_file(
                reduce_covectors(job, [1.0] * job.n_tasks), self.space
            )

        return self._memo("vjp", _digest(dual.values), compute)

    def apply_normal(self, dv: Any) -> ControlVector:
        """Return ``Re(J^H W J) dv`` from Sauce's ``normal`` action (memoized)."""

        direction = self._control_vector(dv)

        def compute() -> ControlVector:
            directory = self._ops_dir()
            job = self._action_job(
                "normal",
                direction=self._direction_stem(direction, directory),
                covector="normal.h5",
            )
            self.problem._run_job(job)
            return _vector_from_file(
                reduce_covectors(job, self.frequency_weights.tolist()), self.space
            )

        return self._memo("normal", _digest(direction.values), compute)
