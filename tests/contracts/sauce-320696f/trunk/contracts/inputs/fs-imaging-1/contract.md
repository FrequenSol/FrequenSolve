# FS Imaging Contract v1

Status: initial
Visibility: public
Schema id: `fs-imaging-1`

## Summary

For workflow selection, support limits and maintained examples, see the
[imaging guide](../../../docs/imaging/README.md).

The imaging contract describes the public JSON block used by imaging workflows to
bind observed data, misfit receiver groups, image grids, and imaging conditions.
The job-level `Imaging` block is still permissive because smoothing and imaging
share setup code; this contract documents the stricter shape expected for forward
plus imaging evaluation cases.

## Versioning

- An imaging block with `"schema": "fs-imaging-1"` is interpreted by this
  contract.
- Imaging-condition kinds are registry names, so the schema accepts custom `IC`
  values in addition to built-ins.
- Preprocessing hooks are versioned hook objects under `preprocess`; supported
  scopes are workflow-level, `misfit`, and individual misfit receiver groups.

## Required Behavior

- `data_path` is the base path for observed receiver data used by misfit receiver
  groups.
- `misfit/receiver_groups` names acquisition receiver groups to compare. Each
  group must provide `name`; optional `observed` and `simulated` values are path
  stems to which the current reader appends the task id and `.h5`. `observed`
  may also be an `HDF5TraceStore`/`SeismicStore` descriptor with `file`,
  optional `dataset`, `missing`, and `source_basis`. For an explicit `df`
  descriptor, `dataset` may name a receiver group in a packed FrequenSolve
  trace root; the reader resolves its active-frequency dataset before reading.
- Observed data `source_basis` is `source_encoding` by default, meaning source
  ids identify encoded RHS fields. When set to `source_geometry`, source ids
  identify physical SourcePoints and misfit forms encoded observed gathers by
  applying the active `Acquisition/source_encoding` weights.
- `misfit/objective` selects the scalar data objective and defaults to `l2` when
  omitted, preserving the legacy objective behavior. Its `kind` accepts
  `l2`/`none`, `huber`/`hybrid`, and
  `studentst`/`student_t`/`student-t`; matching is case-insensitive. Huber uses
  positive `delta`. Student's t uses positive squared scale `c2` and positive
  degrees of freedom `nu`.
- `misfit/comparison` selects the receiver attribute presented to the objective
  and defaults to `{"kind":"waveform"}`. `phase_derivative` compares the
  stabilized physical-frequency phase slope
  `Im(v^H df(v))/(2*pi*(v^H v + epsilon^2))`, in seconds, at fixed Laplace
  damping. Residuals retain the solver convention `observed - simulated`.
- `phase_derivative` requires `observed_derivatives/df` for every receiver
  group. The derivative dataset must identify its base dataset and declare
  `derivative_axis=df`, `frequency_unit=Hz`, derivative units compatible with
  the base channel units per Hz, and the same `source_derivative` policy as the
  comparison. Runtime neighboring-frequency differencing is not a valid
  substitute.
- `source_derivative` defaults to `frozen`, which holds authored source and
  instrument spectra fixed while retaining propagation and model-dependent
  load derivatives. `total` also differentiates authored spectra and is an
  error when those derivatives are unavailable.
- `relative_amplitude_floor` defaults to `0.01`. For each receiver group,
  encoded RHS, and component, epsilon is that fraction of the RMS preprocessed
  observed base amplitude over active receivers. Observed and simulated phase
  attributes use the same model-independent epsilon. A zero-energy component
  contributes zero and the complete objective must contain an active sample.
- Receiver `projection` defaults to `identity`. `up_down` infers acoustic
  reconstruction when the receiver group has compatible pressure and normal-
  velocity channels. It forms the physical upgoing pressure characteristic
  `0.5*(p-Zp*vn)` before comparison and applies the exact transpose
  `0.5*(q,-Zp*q)` to adjoint receiver loads. The deprecated
  `acoustic_upgoing` spelling remains an acoustic-only alias.
- `up_down` instead infers elastic reconstruction when pressure reconstruction
  is unavailable and the receiver group has collocated velocity and vertical-
  traction components plus `shear_impedance`; `physics` may explicitly select
  `acoustic` or `elastic` when both channel sets exist. For vertical normal
  `n=e_z`, the retained elastic traction-like characteristic is
  `0.5*(-sigma*n-Z*v)`, with P impedance for the normal component and S
  impedance for tangential components. The exact transpose is used for adjoint
  receiver loads. Elastic reconstruction therefore requires `velocity_z`,
  `stress_zz`, and every active tangential velocity/vertical-shear pair; the
  component names are configurable.
- `impedance` supplies acoustic or elastic P impedance and
  `shear_impedance` supplies elastic S impedance. Each is either one positive
  scalar, a positive inline value per receiver, or a rank-one real `HDF5Dense`
  dataset with one value per receiver, expressed in aligned traction/velocity
  units. Sparse data require every retained characteristic/velocity pair for
  each active source/receiver key. General receiver-averaging maps require
  scalar impedances; ordinary dense rows and sparse receiver ids accept
  receiver-wise values. Elastic phase comparison remains componentwise after
  this projection.
- Legacy top-level `norm` and the legacy `misfit/norm`, `delta`, `c2`, and `nu`
  fields are deprecated and do not activate a robust objective. Use the
  explicit `misfit/objective` object so behavior, provenance, and restart state
  are unambiguous.
- `grid` is the regular imaging grid passed to `read_grid`.
- `images` lists image products. Each image has optional `name`, `IC`,
  `property`, and `sources`. `sources` is a non-empty list of unique, one-based
  encoded source/RHS ids; when present, only those forward/adjoint pairs
  contribute to that image. This lets one batched imaging job produce separate
  kernels for receiver experiments assigned to distinct encoded RHS ids.
- `keep_unstacked: true` expands each image without an explicit `sources`
  selection into one image per encoded RHS. Names are `source_01_<name>`,
  `source_02_<name>`, etc.; source ids use at least two digits. Explicit
  `sources` selections keep their names and grouping. Frequency stacking and
  smoothing operate on each resulting image independently. With the default
  `false`, images without `sources` average all encoded RHS contributions.
- `IC` defaults to `up_down` in the current seismic builder.
- `save_path` defaults to `ResultPath/imaging/`.
- `post_process`, `keep_forward`, `keep_adjoint`, and `keep_unstacked` default
  to false. `field_retention` is the preferred durable-field policy and accepts
  `none`, `forward`, `adjoint`, or `all`; when present it overrides both legacy
  keep flags. `field_scratch_path` may place disposable field containers on a
  separate shared scratch filesystem; it defaults to `save_path`.
  For `lsrtm_gradient`, `forward` retains both background `forward` and Born
  `inc_forward` fields; `adjoint` retains the residual `inc_adjoint` field;
  `all` retains all three. Retained datasets include every encoded source and
  its RHS normalization, plus material properties, in `fields_<task>.h5`.
  Unretained fields needed for the gradient reuse batch-local transient storage,
  which is removed at normal completion.
  Post-processing currently requires every active physics to be an
  acoustic or elastic DPG formulation. Otherwise initialization warns once and
  disables it for the entire problem. `sync_local_shards` defaults to true, drains closed rank-local
  transient wavefield shards before the next solver stage, and discards clean
  forward-shard cache pages; set it to false to permit deferred writeback.
- `born_traces_only` defaults to false. When true for a `born` workflow, the
  solver writes incremental receiver traces and then stops without storing
  incremental wavefields, solving the incremental adjoint, or writing an
  incremental image. This mode is intended for matrix-free operators that use
  the Born traces as the input to a separate transpose action.
- `gauss_newton` defaults to false for the legacy `born` workflow. The explicit
  `lsrtm_gradient` workflow enables it automatically, evaluates the background
  and incremental forward fields in the same task, forms the complete
  `F0 + Jm - d` residual in Sauce, and writes the incremental image as the
  current LSRTM gradient. A new LSRTM task replaces its existing task kernel
  file, including prior image names, grid metadata and spectral metadata;
  subsequent source batches accumulate into that newly initialized product.
  Legacy Born preserves its existing background `/image` product and replaces
  only `/incremental` on its first source batch. This is replacement, not resume
  or concurrent-writer support. The LSRTM L2 objective is stored on the incremental
  simulated receiver dataset with `objective_role=linearized_residual`.
- `lsrtm_gradient` requires exactly one Cartesian iterate source: `direction`
  names a read-only HDF5 file containing `image/<image-name>` datasets in the
  normal solver image/value-frame representation, or `zero_direction: true`
  explicitly selects the zero iterate. Frequency tasks may safely read the same
  direction file concurrently; solver outputs never overwrite it. The workflow
  deliberately recomputes the background field for each task so randomized
  frequency/source encodings cannot accidentally reuse incompatible state.
- `max_task_field_storage_gib` defaults to zero (unbounded). A positive value
  bounds the estimated new HDF5 chunk allocation for one task before any field
  files are created. `field_storage_limit_action` defaults to `error`; `warn`
  permits the task to proceed after reporting the overage. Workflow schedulers
  remain responsible for multiplying the per-task plan by the maximum number
  of concurrent frequency tasks and enforcing filesystem quotas.
- `reassemble_adjoint` is read by the current top-level imaging setup and defaults
  to false.
- `preprocess` may be an array of `fs-preprocess-hook-1` hook objects or an
  object with `include_defaults` and `hooks`. An explicit empty array disables
  legacy defaults at that scope. Non-empty `images[*].preprocess` remains
  reserved and is rejected by runtime v1.

## Built-In Imaging Conditions

Current built-ins are registered by `register_builtin_ics()`:

- `pressure`
- `velocity`
- `up_down`
- `energy:acoustic`
- `energy:elastic`
- `fwi:acoustic`
- `fwi:acoustic_up_down`
- `fwi:elastic`

Bare `fwi` and `energy` select the active acoustic or elastic variant, preferring
elastic when both physics are active. Explicit kind strings avoid ambiguity.
`fwi` may read `property` to target a material property such as `vp`.

For an exact FWI gradient of the projected upgoing-data objective, use the
`up_down` receiver projection with `fwi:acoustic`. The optional
`fwi:acoustic_up_down` condition additionally restricts the volume kernel to
the physical downgoing source characteristic and physical upgoing receiver
characteristic. It supports pressure-related properties (`vp`, `sp`, `k`,
`beta`, and `qp`); density is rejected because a complete directional density
kernel needs more than the local pressure/vertical-velocity decomposition.

## Data And Output Contract

- Observed data is read from HDF5 receiver files under the receiver group name,
  or from an HDF5 trace store when `observed._type` is `HDF5TraceStore` or
  `SeismicStore`.
- Simulated data defaults to the receiver group's output file unless `simulated`
  provides a path stem.
- Misfit values are accumulated in real64 as a source-batch-invariant `misfit`
  sum on the simulated HDF5 receiver group. A singular objective records
  `objective_kind` and `objective_parameters=[delta,c2,nu]`. For
  `objective_terms`, metadata is indexed by the zero-based configured term
  position using the `objective_term_<index>_` prefix; `id`, `weight`, and
  `value` identify each contribution, and the loss, comparison, and
  preprocessing attributes use the same prefix.
- Durable per-task imaging fields are written to `fields<task>.h5`. Disposable
  fields are written to `fields<task>.transient.h5` and the complete transient
  file is deleted at image-manager teardown. Bulk field payloads are never
  reclaimed by unlinking datasets from a retained HDF5 file, because link
  deletion does not reliably reduce the physical file allocation.
- Image products are written under the image manager output stem and carry grid
  attributes such as `dims`, `x0`, `x1`, and `n_grid`.
- Stacked imaging output writes versioned image groups for raw or incremental
  results.

## Preprocessing Hooks

Each public preprocessing hook object has:

- a versioned hook schema id
- an explicit `kind` and `stage`
- deterministic parameters
- provenance emitted into `_fs_run/.../preprocess.json`
- a shape-preserving rule for v1 buffers

Hooks must not silently mutate observed or simulated receiver data in place.
The built-in v1 hook kinds are `offset_power`, `offset_taper`,
`source_scalar_fit`, `component_scale`, `trace_normalize`, `frequency_weight`,
`source_spectrum_correction`, `trace_mask`, `trace_weight`, `receiver_ar1_whiten`, `scholte_notch`, `slow_velocity_mute`,
`amplitude_clip`, and `wavefield_source_taper`.

Phase-derivative comparison requires a bundle-consistent JVP/VJP. Version 1
accepts frequency-independent linear offset power/taper, component scale,
trace mask/weight, and receiver AR(1) whitening. It also accepts residual-stage
frequency weighting, externally supplied `source_spectrum_correction`, and
dense Scholte/slow-velocity filtering with an explicit spectral-derivative
policy. It rejects `source_scalar_fit`, trace normalization, amplitude clipping,
and frequency weighting in observed or simulated transform stages. The
validation diagnostic names the incompatible hook.

`source_spectrum_correction` is a `simulated` hook for alternating or externally
orchestrated source estimation. `scale` supplies one complex correction or a
global per-RHS array, and `frequency_derivative` supplies the matching physical-Hz
derivative with the same shape. Complex values use `[real, imaginary]` pairs.
For modeled base and derivative traces `(u, u_f)`, the hook produces
`(q*u, q*u_f + q_f*u)` and applies the exact triangular stored-load transpose.
The supplied `q` and `q_f` are held fixed during the model-gradient evaluation;
an orchestrator should therefore update them between outer iterations from a
smooth multi-frequency source estimate. `source_scalar_fit` remains available
for waveform objectives, but its independent per-frequency variable projection
does not define `q_f` and is rejected for phase-derivative comparison.

`offset_taper` and `wavefield_source_taper` use `d0` and `d1` for the taper
limits. With `scale: "absolute"`, these may be scalar values paired with
`units`, for example `{"d0": 0.1, "d1": 0.15, "units": "km"}`, or per-value
objects such as `{"d0": {"value": 100, "units": "m"}}`. With
`scale: "domain_fraction"`, `d0` and `d1` are dimensionless fractions of the
largest domain extent. Source- and receiver-spacing scales are not supported,
so taper widths remain independent of acquisition density. The default
wavefield source taper uses `d0: 0.005` and `d1: 0.01`.

For objective evaluation, hook stages have explicit mathematical roles:

- `observed` hooks transform the fixed observed data before residual formation.
- `simulated` hooks transform modeled data before residual formation and their
  dual actions are applied in reverse order to the robust score.
- `trace_pair` supports at most one `source_scalar_fit` and fixed linear
  receiver operators. Pair hooks are evaluated after observed/simulated
  transforms; their dual actions participate in the same reverse chain.
- `residual` hooks are nonnegative, real objective weights. Built-in
  `offset_power`, `offset_taper`, `component_scale`, `frequency_weight`,
  `trace_mask`, and `trace_weight` hooks are supported in this role. Nonlinear residual transforms
  are rejected instead of producing a mismatched objective and adjoint.

`trace_weight` supplies finite nonnegative real objective weights. Its
`weights` value is normally an `HDF5Dense` reference containing the logically
shaped array; compact inline arrays remain supported for examples and
compatibility. The explicit `layout` is `receiver`, which broadcasts across
sources and components; `component_receiver`, with receiver varying fastest;
`source_component_receiver`, source-major with receiver varying fastest and
indexed by global source id across batches; or `sparse_trace`, which follows the
sparse trace catalog order. The `values` key remains an accepted inline alias
for `weights`. Bulk values are not copied into preprocessing provenance.

`receiver_ar1_whiten` is a fixed linear `trace_pair` operator for stationary
nearest-neighbor receiver covariance. With authored correlation `rho`, the
first ordered receiver is unchanged and subsequent samples are transformed as
`(x_i-rho*x_(i-1))/sqrt(1-rho^2)`. The exact conjugate transpose is applied to
the robust score, so Huber and Student-t objectives remain consistent with the
whitened residual. Correlation must be finite with absolute value below one.
Version 1 requires a complete ordered dense receiver axis and rejects sparse
trace catalogs whose neighbor topology is ambiguous.

`scholte_notch` and `slow_velocity_mute` are frequency-local spatial filters
along a regularly sampled dense receiver cable. Both apply an FFT along cable
order, multiply by a smooth real symmetric receiver-wavenumber mask, and
transform back. They therefore have an exact self-adjoint reverse action.
Receiver coordinates may follow a curved cable, but consecutive along-cable
spacings must agree within `spacing_tolerance`; sparse trace catalogs are
rejected because they do not define a complete regular FFT axis.

`scholte_notch` accepts exactly one ridge center: `phase_velocity`, giving
angular wavenumber `2*pi*frequency/phase_velocity`, or a measured angular
`wavenumber`. `relative_half_width` specifies the zeroed half-width around both
signed ridge branches and `relative_taper_width` specifies its raised-cosine
edge. `slow_velocity_mute` rejects apparent velocities at or below
`stop_velocity`, passes velocities at or above `pass_velocity`, and tapers
smoothly between them. Its `keep_slow` mode uses the complementary mask.
For phase-derivative comparison, `spectral_derivative: "total"` (the default)
includes the analytic physical-Hz derivative of the frequency-dependent mask
in both the bundle JVP and its triangular transpose. `"frozen"` deliberately
sets that mask derivative to zero. A wavenumber-centered Scholte notch has zero
mask derivative under either policy, while a velocity-centered notch and both
slow-velocity modes generally do not.

These filters run after receiver-device evaluation, including DAS gauge
integration. The same forward action is applied to observed and simulated
traces before residual formation and to Born/incremental traces. The score is
then passed through the conjugate transpose, yielding `J^H P^H P r` for the L2
incremental objective without modifying stored raw traces.

`trace_mask` with `invalid: true` is restricted to the fixed `observed` stage;
simulated/residual objective stages require deterministic explicit-id masks.
`trace_ids` require a sparse trace layout, while dense layouts may mask by
source, receiver, or component id.

The objective is a sum over residual samples. With metric-weighted squared
amplitude `q` and residual weight `a`, the supported radial losses are
`a*q/2` for L2, `a*q/2` below the Huber transition and
`a*delta*(sqrt(q)-delta/2)` above it, and
`a*(nu+1)/2*log(1+q/(nu*c2))` for Student's t. The residual stored for adjoint
construction is the corresponding score, including the data metric and
objective weight exactly once.

`source_scalar_fit` uses variable projection with the same data metric,
objective weights, and robust loss as the enclosing objective. Its default
`norm` is inherited; an explicit per-hook norm must be equivalent to the
objective or initialization fails. `max_iterations` and `relative_tolerance`
control robust IRLS convergence (`n_irls` remains a compatibility alias for
`max_iterations`). Fit scalar, iteration count, and convergence status are
written to preprocessing provenance. Robust incremental/Hessian residual mode
is rejected until a baseline score-weight state is supplied. Incremental mode
with source fitting is likewise rejected until a projected baseline
linearization state is supplied; neither case silently falls back to an
inconsistent approximation.

## Diagnostics

Validation and support tooling should surface these failures as structured
diagnostics:

- missing `data_path` for an imaging workflow
- missing or empty `misfit/receiver_groups`
- receiver group name not present in acquisition
- observed/simulated receiver file missing
- missing or inconsistent observed `df` metadata, base-dataset reference,
  physical-Hz units, or source-derivative policy
- unsupported receiver projection or phase-incompatible preprocessing hook
- unsupported objective kind or nonpositive objective parameter
- a source-fit loss that differs from the enclosing objective
- a preprocessing stage/kind combination without a consistent objective/dual
  interpretation
- irregular or sparse receiver geometry requested by a receiver-wavenumber hook
- robust incremental residual evaluation without a baseline score-weight state
- source-fitted incremental evaluation without a projected baseline
  linearization state
- missing `grid`
- empty `images`
- unknown imaging-condition kind
- imaging condition requests a field plan or property that is not registered
- image grid metadata mismatch across forward, adjoint, and property data

## Illumination policy for image smoothing

`Imaging.Smoothing.illumination_normalization` explicitly selects how a stacked
image is preconditioned before the variational smoother is applied:

- `none` smooths the unnormalized adjoint image and therefore preserves the
  linear RTM/Born transpose operator.
- `source` divides by a floored forward-illumination diagonal.  At a fixed
  baseline model and acquisition this diagonal is independent of the residual,
  so the resulting preconditioned adjoint remains linear in receiver data.
- `cross` retains the legacy forward-plus-adjoint illumination normalization.
  Because its denominator includes adjoint illumination, it is nonlinear in
  receiver data and must not be used as an LSRTM/Born normal operator.

The legacy Boolean `normalize_illumination` remains accepted: `false` maps to
`none` and `true` maps to `cross`.  The explicit policy takes precedence when
both keys are present.

## Proximal smoothing operation

`Smoothing.operation` is additive. `riesz` is the unchanged default and keeps
the existing image/control representation-map semantics. `proximal` instead
evaluates

\[
\operatorname{prox}^{D}_{\tau R}(z)=\arg\min_x
\tfrac12\lVert D^{1/2}(x-z)\rVert^2+\tau R(x)
+I_{\rm bounds}(x)+I_{\rm mask}(x).
\]

The proximal form accepts TV or TGV2, a positive scalar or Cartesian
grid-diagonal `fidelity`, physical grid spacing inherited from the image, an
optional active mask with fixed inactive values, scalar lower/upper bounds,
convergence tolerances, and fail-closed or `return_best` nonconvergence policy.
A proximal request is not interpreted as an objective penalty or as a Riesz
map request.


## Compatibility

The job-level schema remains permissive while smoothing and imaging share the
same `Imaging` block. Evaluation validators should apply `fs-imaging-1` only when
the active workflow needs top-level imaging data and misfit setup.

## Elastic density imaging

Cartesian elastic FWI `rho` kernels use the material's declared parameter space. In `velocity-density` and `slowness-density`, density changes both inertia and compliance while the other coordinates are held fixed. In `bulk-shear-density`, `pmod-shear-density`, and `lambda-shear-density`, the stiffness coordinates remain fixed and the constitutive density contribution vanishes. Aggregate images, per-source images, incremental loads, and mixed frequency/material images follow the same convention.
