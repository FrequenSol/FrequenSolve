# FS Job Contract v1

Status: initial
Visibility: public
Schema id: `fs-job-1`

## Summary

See the maintained [imaging guide](../../../docs/imaging/README.md) for imaging
workflow selection, configuration, qualification and known gaps.

A job file defines the workflow, task controls, result path, and simulation
source for a FrequenSolve run. Frequency-domain workflows use `f_list`; the
frequency-independent `raytrace` and `eikonal` workflows use versioned
`RayTracing` and `Eikonal` blocks.

## Versioning

- A job with `"schema": "fs-job-1"` is interpreted by this contract.
- A job without `schema` is treated as legacy v0 by current Fortran readers.
- v1 preserves legacy string simulation paths while adding inline simulation
  objects and file references.

## Required Behavior

- `name` names the task family and contributes to temporary task names.
- `project_path`, when present, defines the project root for this job and takes
  precedence over `simulation/project_path`; explicit command-line and
  environment project-path overrides remain higher precedence.
- `workflow` selects the run mode. Current public values are `forward`,
  `forward_df`, `forward_ds`, `adjoint`, `smooth`, `rtm`, `born`,
  `lsrtm_gradient`, `fwi_operator`, `modal`, `size`, `raytrace`,
  `eikonal`, and `transient`.
- `simulation` may be a legacy path string, an `fs-file-ref-1` object, or an
  inline `fs-simulation-1` object. Legacy relative path strings may be resolved
  under job-level `project_path` when that field is present.
- `result_path` is honored as absolute when it starts with `/` (or a Windows
  drive prefix), even if the directory does not exist yet. Non-absolute paths
  keep the legacy behavior: an existing local path is used as provided,
  otherwise the path is resolved relative to the project path.
- The only command-line result-directory override is `--result-dir`; the job
  JSON field remains `result_path`.
- `f_list` is required by frequency workflows and prohibited for `raytrace`,
  `eikonal`, and `transient`. Current frequency readers
  accept complex values; real numeric values are the common forward path.
- `fwi_operator` receiver actions use `receiver_linearize` to export an
  immutable state, `receiver_jvp` for a material-control tangent, and
  `receiver_vjp` for an independently keyed receiver dual. The linearization
  accepts `field_retention: checkpoint` (default) or `replay`; VJP accepts
  `field_reuse: checkpoint` or `replay` (defaulting to the state policy).
  Checkpoint reuse loads complete saved base/df fields for each active source
  batch and still assembles a fresh operator. These actions require
  full-dimensional 2D or 3D acoustic, elastic, or coupled physical shots,
  Cartesian or sparse receivers, and material controls.
- `workflow: "forward_df"` computes the forward field and its analytic
  derivative with respect to real physical frequency, writing receiver datasets
  with the ordinary name and the `_df` suffix. `workflow: "forward_ds"` does
  the same for imaginary physical frequency and the `_ds` suffix. Each workflow
  computes only its selected derivative direction. `derivative_order` defaults
  to one and accepts orders through four. The solver writes every order through
  the requested maximum using `_df`, `_d2f`, `_d3f`, and `_d4f` (or the
  corresponding `s` suffixes). Higher derivatives use a raw-derivative
  recurrence with the frozen DPG optimal-test map and retain only one extra
  condensed right-hand side while the full DOF buffer is reused in place.
  Orders above one currently require DPG. Their scope is standard acoustic,
  elastic, poroelastic, or compiled Maxwell physics, plus coupled acoustic-elastic and
  acoustic-elastic-poroelastic DPG, with native receiver variables. The DPG
  optimal-test map is frozen. For acoustic, elastic, and poroelastic physics,
  artificial impedance and point-source spectra are frozen. Maxwell differentiates
  its impedance admittance, conductive and cold-plasma material dispersion, and
  Cartesian PML tensors. Maxwell supports 2D, 3D, and 2.5D at fixed transverse
  wavenumber or toroidal mode. Its incident-field derivative supports isotropic
  `MaxwellPlaneWave` and vacuum `cylindrical_bessel` impedance sources; Gaussian
  beams and anisotropic incident eigenmodes are rejected. Maxwell phase-objective
  adjoint gradients remain unsupported. Maxwell recurrence loads use per-source
  normalization estimated from the previous solution magnitude and frequency;
  receiver value conventions restore the raw physical-frequency derivatives.
  Acoustic and elastic PML stretching is differentiated; poroelastic PML
  transforms remain frozen. Kjartansson constant-Q material dispersion is differentiated for
  acoustic, ISO/TI/TTI elastic, and direct or ISO/VTI/TTI-frame poroelastic
  materials; poroelastic Darcy/JKD frequency dependence is also differentiated. DPG gravity surfaces are
  differentiated, including time-domain external pressure and frequency-domain
  pressure with the selected explicit derivative datasets. A frequency-domain
  field missing a derivative required by the requested order is rejected. Attenuative tangents require nonzero
  frequency. `Discretization/DPG_trial_to_test` in the selected simulation
  controls whether the base-frequency DPG trial-to-test action is stored or
  recomputed element by element for the tangent load. The former
  `frequency_tangents` job object is not accepted.
- `workflow: "raytrace"` requires a job-level `RayTracing` object satisfying
  `fs-ray-tracing-1` and prohibits `f_list`. Ray tracing is not dispatched as a
  synthetic zero-frequency task.
- `workflow: "eikonal"` requires a job-level `Eikonal` object satisfying
  `fs-eikonal-1`, prohibits `f_list`, and runs the real64 native-mesh acoustic
  first-arrival solver without constructing FEM degrees of freedom.
- `workflow: "transient"` requires a job-level `Time` object satisfying
  `fs-time-domain-1` and runs as one task without `f_list`. The first public
  slice supports full-dimensional Cartesian Galerkin acoustic, elastic, and
  coupled acoustic-elastic waves with Newmark average acceleration, plus
  pressure-form Galerkin Darcy flow and temperature-form thermal diffusion with backward Euler. Receiver and
  configured wavefield outputs are written as deterministic per-step HDF5
  shards, and optional rank-local checkpoints can resume the same fixed
  interval.
- Job files must not define `f_adapt`. Mesh adaptivity frequency controls now
  live in `simulation/Mesh/adapt` as `f_low`, `f_high`, and `f_adapt`.
- `k_list` is required for half-dimension workflows. `k_weights` may provide
  matching quadrature weights; otherwise current readers compute trapezoid
  weights from sorted signed `k_list`. `k_units` defaults to `1/m`.
- `Imaging` is required by imaging workflows and smooth runs. Forward plus imaging
  evaluation cases should satisfy `inputs/fs-imaging-1`; smooth-only runs may use
  a narrower subset while the shared reader remains permissive.
- `Modal` is required by the `modal` workflow. It defines an elliptic contour,
  a range of physical source-encoding fields used as probes, a configurable
  simultaneous-RHS limit, initial quadrature order, adaptive-refinement and
  residual tolerances, numerical-rank tolerance, and an optional retained-mode cap.
  Its `center` defaults to and, when supplied, must
  match the active `f_list` entry so initialization and every contour solve use
  one fixed reference discretization. A zero `probe_count` uses up to 32 available
  source probes independently of `rhs_batch_size`. Beyn reduction uses recovered L2 field coefficients;
  trace unknowns are not included. Verified extraction doubles `quadrature_order`
  until successive pole sets stabilize or `max_refinements` is exhausted.
  `require_operator_residual` cannot pass with the current field-only seismic
  adapter because a true DPG residual also needs trace unknowns. Saturated probe rank and unverified pole sets
  are reported explicitly; the DPG response family remains experimental and is
  not certified analytic in complex frequency.
- `lsrtm_gradient` is the Cartesian linearized-residual workflow. For every
  frequency task it solves the background forward problem, solves the Born
  forward problem for the supplied Cartesian image direction, forms
  `F0 + Jm - d` and its L2 objective inside the executable, and solves one
  incremental adjoint to write the batch gradient. It requires observed data
  for every configured misfit receiver group and does not use native
  `control_sensitivities`.
- `fwi_operator` exposes an immutable, task-local objective linearization.
  `linearize` writes version-3 distributed objective state and an optional
  gradient. `jvp`, `vjp`, and `normal` use one ordered `controls.active`
  subspace and canonical `direction`/`covector` files. Joint `normal` computes
  `J*J`, including cross-block terms, with saved weights, phase floors and masks.
  Every action validates the frozen state, control registry and MPI partition.
  Complex controls use real-interleaved coordinates and the real-Hermitian pairing.
  The `wri` action instead solves one reduced wavefield-reconstruction
  subproblem per source-encoding RHS,
  `min_u 0.5 ||B(m)u-q||^2_G^-1 + 0.5 lambda ||Pu-d||^2_s^-2`.
  `wri/objective_normalization` defaults to `observed_energy`: all reported
  objective terms, model covectors and curvature products are divided by
  `lambda * ||S^-1 d||^2`, using the same observation selection, projection,
  observed preprocessing and residual weights, summed across source batches.
  This common factor does not alter reconstruction. `none` disables it; a
  positive number specifies a fixed divisor. Zero observed energy is rejected.
  The artifact saves `/observed_energy` and `/objective_normalization` with a
  `policy` attribute. Multi-frequency reduction uses the ratio of weighted sums,
  not a sum of independently normalized values. `wri/normalization_only: true`
  exports the observed calibration without assembly, solves, covectors or
  `/value`; it excludes curvature. Its full-survey divisor can be frozen as a
  numeric `objective_normalization` throughout inversion.
  The positive `wri/penalty` is `lambda`; `wri/data_scale` resolves to a scale in each
  selected receiver component's coordinate units (unit-aware `auto` by default). The observation term is
  inserted into the uncondensed DPG normal system before bubble condensation,
  so the reconstructed field uses the same coupling and static-condensation
  path as an ordinary solve. When `control_sensitivities` is present, Sauce
  uses the envelope theorem to write the reduced model covector from local
  PDE-residual contractions; no wavefield adjoint solve is required.
  `wri/formulation` is `centered` (default) or `original`. Centered subtracts
  the current model's minimum PDE energy and its model covector, using one
  ordinary forward reference solve per source batch. Original retains the
  uncentered broken-test residual energy without the reference solve. Both
  reconstruct the same wavefield. `/value` records its `formulation` attribute;
  `/pde_objective` records its `definition`, and centered output additionally
  saves the removed `/reference_pde_objective`. Existing curvature products
  remain the original positive GN surrogates, not centered-objective Hessians.
  At a receiver point incident on multiple broken DPG elements, Sauce applies
  the penalty to every incidence with weights summing to one. This keeps the
  assembled operator and reported objective identical while also discouraging
  disagreement between reconstructed traces at the observation boundary.
  This implementation supports full-dimensional acoustic, classic elastic and
  Maxwell DPG, multiple waveform least-squares receiver groups, dense or sparse point
  receivers, identity or acoustic/elastic `up_down` projections, and point-local
  preprocessing. Select groups with `receiver_groups`, select one with
  `receiver_group`, or omit both to use all imaging groups. The selectors are
  mutually exclusive; selected names must be unique and resolve to imaging
  misfits. Each group uses its own observed data, projection, and pipeline.
  Observed-stage hooks operate on fixed data. Simulated stages must be fixed
  linear local operations; residual stages supply finite nonnegative objective
  weights. Projection precedes preprocessing in receiver coordinate units, then
  WRI component scales are applied. Spatial trace-pair filters, postprocessed
  fields, and sample averaging remain unsupported. FrequenSolve remains responsible for selecting and updating
  the penalty, alternating or variable-projection model/source updates, and
  continuation across tasks.

  With `formulation: original`, source-independent dense WRI observations share
  the reconstruction matrix and factorization across source batches, including
  a shorter final batch. Centered WRI retains the hierarchy but refreshes its
  operator and factors between each batch's reference and reconstruction solves.
  Sparse layouts and potentially source-dependent preprocessing error before solver
  setup if any configured batch has multiple RHSs. Batch sizes are never reduced
  automatically. Explicit single-RHS batches remain supported and rebuild per source.
  Receiver/component-only masks and trace weights preserve dense factor reuse.
  Observed group/component `trace_normalize` requires one complete source batch;
  multiple batches are rejected to avoid batch-dependent normalization.
  Missing sparse rows impose no observation penalty. Sparse sampling weights
  are retained. The objective artifact's `/data_scales` concatenates component
  scales and labels each with its `receiver_groups` attribute. PDE and data objective terms and model
  covectors undo numerical RHS normalization, so numerical source scaling does
  not change the relative penalty. This corrects the earlier single-RHS path,
  whose data/PDE balance depended on that normalization. Existing WRI penalty
  choices may therefore require retuning. The new default `data_scale: auto`
  (including omission) uses one solver unit per component, expressed in that
  component's coordinate units. This is independent of sources and observations.
  A positive number supplies an explicit coordinate-unit scale; extreme
  data/PDE weight ratios can still be ill-conditioned. Existing jobs that omitted
  the field change behavior; explicit `data_scale: 1` restores that scale choice.
  The objective artifact always writes `/data_scales`, with `components`,
  `units`, and `policy` attributes (`auto_solver_units` or `explicit`).
  The legacy scalar `/data_scale` is written only for explicit numeric input.

  Waveform and full-complex spectral `linearize` may request `receiver_diagonal: {}` to write
  `receiver_diagonal_<task>.h5`; optional `probes`, `seed`, and `output` configure
  the shared global receiver encoding. The default count is min(16, encoded
  source RHS count), not a source-batch limit. See the supported layouts and
  control-basis contraction in [curvature](../../../docs/imaging/curvature.md).

  Spectral derivative RTM also accepts `control_sensitivities.receiver_diagonal`.
  With `kernel_derivative.export_lower_orders`, matching `receiver_diagonal_dN`
  artifacts accompany lower-order gradients; the highest order keeps the ordinary
  output name. One probe hierarchy supplies orders 0–4 when receiver metrics differ
  only by an order scalar. Observed normalization is included independently for
  each order. Probe RHS width does not alter source batching. `--smooth` aggregates
  these diagonals with the gradient frequency weights but does not smooth them.

  Optional `wri/diagonal: "diagonal.h5"` writes the fixed-wavefield frozen-Gram
  GN diagonal alongside the gradient, in the same control coordinates and
  objective normalization. It excludes curvature actions and normalization-only
  runs. Local control-basis contractions are squared after their full element
  integration; no additional global solve is required.

  A `model_direction` changes `model_covector` to a model-normal action.
  Centered WRI defaults to `wri/curvature: metric_frozen`, a positive centered
  approximation with two extra solves. `exact_gn` applies the full centered
  residual GN normal with four extra solves. Both include the nonlocal reference
  response and full normal-equation derivative. Original WRI defaults to
  `joint_schur`. Explicit `fixed_wavefield | joint_schur` retain the uncentered
  approximations; they are not centered GN. The former
  holds the reconstructed field fixed; the latter eliminates its increment
  from the joint Gauss–Newton system. All modes retain cross-parameter entries and
  require frozen Gram weights and unwindowed material controls in acoustic,
  classic elastic, coupled acoustic–elastic or Maxwell DPG. They are
  positive-semidefinite approximations, not exact reduced Hessians. Relaxed
  assembly is accepted as a further approximation; exact assembly
  (`Solver/relaxed_assembly=false`) keeps the reconstruction and the curvature
  contractions on the same Gram factor. The objective output still describes the base
  reconstruction; its `value` dataset records `curvature` and `gram_derivative`
  attributes. WRI reconstructs the base state within each invocation; these
  actions do not consume a waveform/phase saved-linearization artifact.
  Coupled acoustic–elastic reconstruction and `pde_objective` include the same
  registered normal-velocity and traction-continuity penalties. These interface
  coefficients have no explicit material derivative. Objective gradients honor
  `gram_derivative: total` in both acoustic and classic elastic domains; the
  curvature actions remain frozen-Gram approximations. See the
  [coupled WRI example](examples/fwi-operator-wri-coupled.json), which assumes
  the referenced simulation defines the listed fluid and solid material controls.
  See [WRI curvature and costs](../../../docs/imaging/wri.md).
- `control_sensitivities.quadrature` defaults to `auto`: unweighted tensor-node
  volume sensitivities for RTM/FWI pullbacks, with native quadrature for other
  controls, unsupported tensor layers, `fwi_operator.extension`, intersected
  assembly, jobs with active geometry controls, and discrete JVP/normal actions.
  Use explicit `wavefield` for exact
  coefficient derivatives and transpose tests.
  Explicit `material_intersections` subdivides volume material pullbacks at material-cell
  boundaries; assembly, face rules and JVPs remain unchanged. These covectors
  approximate continuous sensitivities rather than the exact discrete objective
  derivative. RTM and FWI `linearize`, `vjp`, `receiver_vjp`, and `wri` accept
  the option, including waveform and spectral receiver-probe diagonal contractions.
  These diagonals are positive preconditioner approximations, not exact discrete
  GN diagonals. Born/JVP, normals and WRI curvature/diagonals reject it.
- `control_sensitivities` selects native material-control sensitivities instead
  of a Cartesian image for a `born` or `rtm` workflow. `born` requires a
  `direction` HDF5 file for its JVP; `rtm` requires a `gradient` HDF5 output path
  for its VJP. An optional `objective` HDF5 path makes the same RTM invocation
  write the robust scalar data objective evaluated before its retained-state
  adjoint solve. Each RTM frequency task writes `gradient_<task>.h5` and, when
  requested, `objective_<task>.h5`; the standard
  `--smooth` postprocess applies optional nonnegative `weights`, sums those
  coefficient covectors and scalar objectives, writes `raw_gradient` (or
  `gradient_raw.h5`), and writes the final `gradient`. Optional `Smoothing` applies a
  representation-owned Tikhonov, TV, or second-order TGV variational regularization solve
  after aggregation. The parts may be native `/controls/<block>` files or
  `fwi_operator` covectors (`fs-control-vector-1`, qualified
  `/controls/model.<block>` datasets); the reader accepts both layouts, ignores
  blocks outside `active`, and rejects an unsupported `/schema` or `/packing`.
  Joint sources produce joint outputs: model blocks are written as
  `/controls/model.<block>`, the remaining joint blocks (source, reflectivity)
  are summed with the same weights, the `/support` masks are copied from the
  first source, and `/schema`, `/packing`, `/state_fingerprint` and
  `/control_registry_fingerprint` are carried through (every part must share
  both fingerprints), so the processed vector can be used directly as
  `fwi_operator.direction`. Native sources produce native outputs.
  Optional `input` names one such vector explicitly: the postprocess then reads
  that file instead of the `_<task>` parts, copies it to `raw_gradient`, and
  writes the smoothed `gradient`; `weights` and `objective` aggregation
  are ignored. `f_list` remains required because wavelength-relative smoothing
  scales use its largest frequency. Relative `gradient`, `raw_gradient` and
  `objective` paths resolve under the job's result
  directory, like the `fwi_operator` outputs, so the same relative string names
  both the `fwi_operator.covector` parts and the smoothing input; a relative
  `input` is looked up there first and otherwise like `fwi_operator.direction`.
  `input_role: dual` is the default because a native VJP is already a weak
  coefficient-space load; `primal` mass-weights supplied nodal values before
  solving. `lambda` multiplies the conservative P inverse wavenumber
  `wavelength/(2*pi)` of the control's owning material layer at the largest
  frequency in `f_list`;
  `alpha` may override the resulting Tikhonov/TV variational coefficient
  directly, and `reference_wavelength` may override only the model-derived
  wavelength. For TGV, the wavelength-scaled defaults are
  `alpha1 = lambda*wavelength/(2*pi)` and
  `alpha2 = tgv_ratio*alpha1^2`; explicit `alpha1` and `alpha2` may replace both
  defaults together. Native control smoothing currently requires a
  MUMPS-enabled build. The optional
  `current` HDF5 file replaces the material model's
  configured control coefficients before the baseline operator is assembled.
  Optional `active` lists the model-parameter blocks included in the packed
  direction and gradient vectors; omitted `active` selects all registered
  blocks. Parameters outside that subspace remain fixed at their current values.
  Optional `spatial_window` restricts native JVP and VJP sensitivity
  contractions to one physical `x`, `y`, or `z` interval while leaving the
  baseline forward fields and receiver residuals unchanged. `minimum` and
  `maximum` define the closed interval; `taper` defaults to zero and otherwise
  gives the raised-cosine transition distance at both edges, so it cannot
  exceed half the interval width. `units` defaults to metres. The same real
  weight is applied to volume, model-dependent source, and impedance terms in
  the Born and transpose paths.

  Native Galerkin actions now include uncoupled acoustic primary-property and
  elastic material controls, including model-dependent Robin/impedance terms;
  axisymmetric acoustic Galerkin is supported. TTI `theta`/`phi` controls use
  analytic angular derivatives. Phase actions include mixed material/frequency
  terms for acoustic and elastic Galerkin volumes and classic/weak-symmetry
  coupled DPG. Remaining formulation and boundary restrictions are listed in
  [the derivative capability guide](../../../docs/imaging/derivatives.md).
  [Multiparameter curvature diagnostics](../../../docs/imaging/curvature.md)
  operate on the saved `normal` action without choosing an optimizer.

  Cartesian Maxwell DPG supports ordinary material Born/JVP, RTM/VJP, and
  shared `fwi_operator` `linearize`, `jvp`, `vjp`, and `normal` actions.
  Geophysical properties, localized controls and complex physical-current
  signatures use the shared registry and versioned control/objective artifacts;
  see [EM workflows](../../../docs/imaging/em-geophysics.md). `normal` is a
  Gauss–Newton/positive-IRLS action, not a full Newton or source-eliminated
  Hessian. Saved operators reject nonlinear `source_scalar_fit` preprocessing;
  the legacy RTM calibration path remains separate.

  Maxwell total derivatives include fixed-stretch PML constitutive pullbacks
  wherever authored controls have support. Impedance admittance and supported
  material-dependent isotropic incident fields are differentiated. Ordinary
  current-source receiver derivatives are qualified for volume E/H channels.
  WRI has a separate receiver and curvature policy. Modal, cylindrical,
  geometry, phase-objective and source-taper extensions are not enabled here.

  `gram_derivative` defaults to `frozen`, preserving the existing approximate
  DPG derivative and runtime cost. Opt in with `total` for ordinary acoustic
  DPG Born/RTM or `fwi_operator` actions `linearize`, `jvp`, `vjp`, `normal`, and `wri`. The WRI
  derivative includes `-0.5 Re(v^H dG v)`, where `v=G^-1(l-Bu)`. Ordinary
  RTM/FWI additionally differentiates the trial-to-test map, including both
  `dB^H v` and the bilinear Gram correction. These are analytic contractions
  of graph-norm coefficients, not finite-difference element kernels or
  explicitly formed inverse derivatives. WRI reuses its existing optimal
  residual; RTM/FWI caches an additional element-local optimal residual and
  currently factors/solves its Gram matrix once per source batch. The frozen
  path performs none of this additional work. With relaxed assembly, total is an
  approximation: the fast-mode assembly can be inconsistent with the separately
  evaluated residual objective. For verification use
  `Solver/relaxed_assembly=false`, fp64 Schur storage and a tight solve tolerance. Both RTM and WRI keep
  frozen Gram as their default; total is an opt-in verification mode.
  Total requires full-dimensional Cartesian acoustic, classic elastic or
  Maxwell DPG material controls; Galerkin, weak-symmetry elasticity, geometry,
  phase derivatives, source tapers and spatial windows are rejected.
  WRI normals still require frozen Gram. Keep the mesh, polynomial orders,
  quadrature, PML stretch and material-extension geometry fixed during checks.
  Seismic and WRI model gradients exclude PML elements; ordinary Maxwell
  includes its fixed-stretch constitutive pullback. Model-dependent refinement
  is not differentiated. Smoothing remains a postprocessing step, not part of
  the raw objective covector. Qualification remains combination-specific; see
  [capabilities](../../../docs/imaging/capabilities.md).

- Cartesian image `Smoothing` accepts `tikhonov`/`l2`, `tv`, and `tgv`/`tgv2`.
  The TGV2 image smoother uses the mixed first-order form
  `alpha1*|grad(m)-w| + alpha2*|sym(grad(w))|` with scalar `m` and an auxiliary
  H1 vector `w`. It therefore requires only first derivatives of ordinary H1
  finite-element fields; it does not require a Hessian, an H2 space, or C1
  elements. `derivative_order` is consequently fixed at one for the image
  path. With wavelength-relative scaling,
  `alpha1=lambda*wavelength/(2*pi)` and
  `alpha2=tgv_ratio*alpha1^2`; an explicit `alpha1`/`alpha2` pair overrides
  those defaults. `epsilon` controls the fixed split-Bregman penalties, and
  `iterations` controls the nonlinear shrink/update count.


## Compatibility

This schema is intentionally permissive through `additionalProperties` because
job-level imaging and smoothing blocks are still read by subsystem-specific
Fortran code. The versioned `RayTracing` subtree is strict even though the job
container remains permissive.

A `forward` job may supply `control_sensitivities/current` to load a named
coefficient checkpoint before solution adaptation. This updates the nonlinear
material state without enabling Born or adjoint sensitivity assembly. Mesh
coefficient datasets must match the frozen property-space identity; they are
read as locally required slices.

The Cartesian 2.5D Maxwell workflow uses the existing signed `k_list` quadrature.
A simulation selecting cylindrical EM with `toroidal_mode` instead solves one
discrete toroidal harmonic and does not consume `k_list` or `k_weights`.


## Total spectral kernel derivatives

An `rtm` or `fwi_operator` job may add `"kernel_derivative": {"order": 4, "axis": "fourier"}`.

Retained spectral recurrence and forward trial fields default to
`kernel_derivative.field_storage: "auto"`. After the base solve, Sauce predicts
`n` recurrence snapshots plus `n+1` element-local forward trial planes from the
actual DOF layout, precision and RHS batch size. This does not assemble forms.
If that payload exceeds the remaining memory budget, all ranks use disk.
`field_memory_fraction` defaults to `0.8`, leaving 20% headroom; zero selects disk
for any nonempty hierarchy. Available host memory (and visible Linux cgroup
limits under standard mounts) is conservatively divided among ranks on each host.
Unknown availability selects disk. Explicit `"memory"` and `"disk"` override auto.
The estimate, summed rank budgets and selected mode appear in result diagnostics
as `spectral_estimated_field_bytes`, `spectral_field_budget_bytes` and
`spectral_fields_on_disk`. These are snapshots, not reservations against other jobs.
Disk mode writes
immutable fields to rank-local scratch files under `--tmp-directory` and maps
them read-only. Pages are loaded on access and are reclaimable by the operating
system; this is not a fixed process-RSS limit. The active solve, solver factors,
receiver data and optional probe caches still need memory. Forward trial packing
can temporarily retain one full plane. No field compression or precision change
is applied. Prefer a local SSD with enough capacity for the retained hierarchy.
Scratch files are unlinked immediately and reclaimed when their mappings close,
including after process termination; they are not restart checkpoints.
`spectral_forward_cache_bytes` and `spectral_recurrence_cache_bytes` report heap
payloads; the corresponding `spectral_forward_disk_bytes` and
`spectral_recurrence_disk_bytes` report mapped file payloads, not physical RSS.
This storage choice does not change the saved physical-state fingerprint.

`order` defaults to one and accepts zero through four; `axis` defaults to
`fourier` (real physical Hz), or may be `laplace` (imaginary physical Hz).
The job requires `Image` with acoustic `fwi:acoustic`, property `vp`, an empty
image preprocessing pipeline, and field retention `none` or `all`.

Sauce solves both field hierarchies d0 through dn, differentiates the acoustic
material prefactor, and writes the complete product-rule derivative of the
physical Vp sensitivity kernel. All orders are raw derivatives, without
factorial normalization. One base operator is assembled per frequency and
reused for `2*(order+1)` solves per source batch. The ordinary forward traces
and misfit are evaluated once per batch.

The source signatures and source/receiver encoding weights are frozen at the
base frequency, as is the receiver residual driving the adjoint hierarchy.
Thus this is a spectral kernel derivative with a fixed data dual. It is not
by itself the gradient of a derivative-data objective or the spectral
derivative of a full FWI gradient with a frequency-dependent residual.
The DPG optimal-test map and artificial impedance also remain frozen, as in
`forward_df`. This option is separate from `control_sensitivities/gram_derivative`.

`residual` selects the data dual (default `"base"`, as above):

- `"derivative"` compares the raw order-n simulated traces with order-n observed
  traces: a `t^n` weighting of the damped traces.
- `"window"` with `"window": [a0, ..., an]` compares `sum_k a_k t^k` weighted
  traces (t in seconds) with one common residual; the polynomial degree sets
  the order.
  Alternatively, with explicit `order`, map every objective receiver-group name
  to an HDF5 dataset locator. The finite real coefficient arrays have Python/HDF5
  shape `(order+1, global_receiver, source_field)` in solver source-field order,
  shared across components. Here `order` is the maximum retained polynomial power;
  individual traces or coefficient planes may be zero.
  These FWI windows require dense point receivers without averaging or
  frequency/material-dependent receiver operators; WRI and probe diagonals are
  not supported. Coefficient inputs must remain fixed with the objective state.
- `"jet"` sums one l2 misfit per order, weighted by `"order_weights"`
  (default ones, highest positive). It applies the triangular transpose of the
  derivative recurrence by adding each order's scaled receiver adjoint to the
  matching plane of the upward adjoint recurrence.

In every mode the forward hierarchy samples the needed dn fields into
`<group>_dnf` or `<group>_dns` datasets (`_df`/`_ds` at first order), the
product-rule accumulator is unchanged, and the image is the exact gradient of
the selected objective up to the frozen quantities above. Every mode costs
`2*(order+1)` solves: a jet folds each order's weighted receiver covector into
the single adjoint solve of its plane. A time-domain
SeismicStore supplies observed `t^k` moments directly; an HDF5 trace file or
packed trace root must provide each needed derivative group.
For separate per-order gradients, control-sensitivity RTM accepts
`kernel_derivative/export_lower_orders: true` with `residual: derivative`.
The primary outputs represent `order`; lower-order gradient and objective paths
add `_dK` before `.h5`, before any task suffix. Postprocessing independently
aggregates all orders. This reuses one forward hierarchy and factorization, but
each objective has its own adjoint recurrence: `(n+1)*(n+4)/2` solves for orders
0–n (20 at n=4). It does not produce multiple `fwi_operator` states and is not
supported by image-kernel, window, jet, or WRI jobs.
With `control_sensitivities` and a `derivative`, `window` or `jet` residual the
job writes the native control gradient and objective of that time-weighted
misfit instead of an image, using the exact discrete transpose of the
recurrence and analytic mixed frequency/material partials through order four
(acoustic DPG, frozen Gram, unstretched cells). `workflow: fwi_operator`
accepts the same key with a `derivative` or `window` residual and a waveform
comparison for `linearize`, `jvp`, `vjp` and `normal`; `jet` is rejected there.
For `action: wri`, `derivative` and `window` select centered windowed-reference
WRI: the reference and observed traces receive the same spectral combination,
while the PDE correction retains the base-frequency energy metric. It requires
`formulation: centered` and fixed, frequency/material-independent receiver rows.
Material gradients accept frozen or total Gram dependence; the spectral recurrence
does not differentiate Gram in frequency. Original WRI, spectral normals and diagonal outputs are rejected.
Cost is `order+2` solves for the objective/reconstruction and `2*(order+1)` with
the gradient, per source batch. See [windowed WRI](../../../docs/imaging/wri.md#spectral-windows).
`"source_derivative": "total"` includes the point-source frequency derivative
in the forward hierarchy, matching `forward_df` and recorded time moments.
First-order spectral Born and normal actions also support this source policy,
including the mixed material/source-frequency load. Higher-order Born actions
require a frozen source spectrum. Gram derivatives remain frozen.

The initial scope is Cartesian 2D/3D acoustic DPG, nonzero complex frequency,
point sources, native receivers, and lossless or Kjartansson material response.
PML propagation is differentiated. Projection (and WRI except for the separate windowed-reference action above), control sensitivities with
a `base` residual, phase-derivative comparisons, incident/equivalent sources, and gravity-surface
forcing are rejected. Builds must support parallel HDF5, including serial runs.
See the [native workflow guide](../../../docs/imaging/spectral-kernels.md)
and [job example](examples/total-kernel-derivative.json).

### Shared inversion state and source controls

The required `fwi_operator/controls/active` list selects ordered blocks named
`model.<control ID>` and `source.<physical ID>.<position|mechanism|signature|signature_df>`.
Unknown, repeated and unsupported blocks fail. `controls/state` optionally
applies a complete canonical baseline before assembly; `controls/state_output`
exports that baseline. `controls/manifest` exports the resolved registry and all
MPI ownership descriptors. Both output paths are exact in a single-task job and
receive the `_<task>` suffix before the extension when `f_list` has more than one
entry. A `state_output` records each mechanism block's physical scaling, so a
baseline written by one frequency task replays in another
([fs-control-state-1](../fs-control-state-1/contract.md)).

The internal, experimental `controls/pml_stage` object supplies `manifest` and
`identity` for an immutable [patch stage bundle](../../internal/fs-patch-stage-1/contract.md).
It requires `controls/state` as the candidate. The runtime verifies the pinned
material definition and context and restores the model baseline before initial
mesh and PML sizing. After acquisition initialization, it validates and restores
the complete stage baseline, captures the PML owner, then applies the candidate.
Stage identity participates in native state reuse checks. Reference-window refresh
verifies the pinned inputs and reloads both material owners. Acoustic/coupled
sizing queries select frozen materials in PML. Controlled patch PML remains gated
pending stage geometry/mesh and application derivative integration.

Operator inputs `state` (jvp, vjp, normal), `direction`, `objective_vector`
(vjp), `extension/direction` and `model_direction` are resolved per task when
`f_list` has more than one entry: the task-suffixed sibling `<stem>_<task><ext>`
is probed first, as an absolute path or as a regular file under the project path
(and, for inputs that are searched, as given). A missing sibling never fails the
job: the exact path is then resolved as in a single-task job. Single-task jobs
use the exact path. `controls/state` is always exact. A saved linearization
records its `task`; another task index rejects it.

`state_output` and `covector` files also carry one packed support bitmask per
block, `/support/<qualified block>`, from the frozen-baseline measure
`s_i = sum_q w_q |dm/dc_i(x_q)|` over the volume quadrature Sauce visits during
material interpolation. `controls/min_support` (default `0.01`) marks a DOF
unsupported when `s_i` is below that fraction of the block's median nonzero
measure; `controls/support_measure: true` adds `/support_measure/<block>`.
Both keys apply to `fwi_operator` only; the native `control_sensitivities`
workflow does not export support. See
[Shared inversion controls](../../../docs/imaging/controls.md#control-support).

Use `direction` for JVP/normal and `covector` for gradient/VJP/normal. The shared
HDF5 coordinate and identity contract is documented in
[Shared inversion controls](../../../docs/imaging/controls.md).
Objective state/vector version 3 supports MPI reload on the same realized mesh
partition. Legacy provider/joint vector fields and objective v1/v2 are removed
from the FWI interface. WRI retains its separate material contract above.

`source_controls` defines only the location derivative policy: `analytic` (default).
The deprecated `local_fd4` spelling is accepted as an alias for `analytic`; it no
longer enables finite differences. `reference_step` remains accepted with its
legacy default 1e-4 and range [1e-6,1e-2], but is unused. Unsupported analytic
basis or coefficient derivatives fail explicitly. Source positions remain fixed
inside an action. Candidate coordinates and mechanisms require a new baseline.
See [source controls](../../../docs/imaging/sources.md) for coordinates,
discretization capabilities and geometric restrictions. FrequenSolve orchestrates
state updates and optimization between calls.

### Auxiliary model extension

`fwi_operator/extension` selects `linearize`, `jvp`, `vjp`, `normal`, or `solve`
in an auxiliary physical-property space. Omit `control_sensitivities`. Each field
names a direct parameterized material property by its unqualified `control` ID
and defines exactly one axis:

- `lags`: uniform `count`, `origin`, `spacing`, and explicit time `units`.
- `offsets`: `half_offsets` as an array of spatial coordinate vectors and explicit
  length `units`. Each vector is half the separation between incident and test
  samples. `packet_mb` bounds native donor exchange. The midpoint halo artifact
  is inferred from the named mesh property space; `artifact` can override it.

Real and complex physical frequencies use the existing job `f_list`: for example,
`[[3.0, -2.0]]` represents `3-2i` Hz. At lag `tau` seconds, the factor is
`exp(-i*2*pi*f*tau)`, including `exp(2*pi*Im(f)*tau)`. Real tap coordinates are
retained; transpose actions conjugate the complete factor, including its amplitude.
This applies to extension JVP/VJP, normals, inner solves, reduced gradients and
reduced Schur actions. Shared-band L2 fits are selected as described below.
Large imaginary-frequency/lag products can impair conditioning; nonfinite
frequencies and factors not representable in the solver working precision are
rejected. Full-dimensional waveform comparisons remain required; relaxed
assembly is accepted as an approximation. See the
[complex-frequency lag example](examples/fwi-operator-extension-complex.json).

Fields borrow spatial control maps, including meshed properties. Tap values use
the property catalog units without nonlinear model transforms or bounds. The lag
sum absorbs quadrature weights. Auxiliary inputs and outputs use
`extension/direction` and `extension/covector` with
[fs-extension-vector-1](../../outputs/fs-extension-vector-1/contract.md).
`extension/manifest` exports global spatial counts, physical axis coordinates and
basis/baseline identities. Linearize may return the residual extension covector.
Output filenames receive the usual frequency-task suffix; relative paths use
ResultPath. `extension/direction` follows the per-task input resolution of the
shared control inputs above. The ordinary `state` and `objective_vector` formats
remain unchanged.

`solve` requires `extension/solver` with positive `damping`, `solution`, and
`report` paths. It solves one regularized quadratic shared across source batches
of this frequency task, or across all frequencies with `frequency_weights`.
`field_scales` supplies one physical amplitude per field:
taps equal scale times the Krylov coordinate. The normal receives damping squared
plus the applicable squared axis penalty: `lag_penalty*tau/lag_scale` or
`offset_penalty*|half_offset|/offset_scale`. A nonzero axis penalty requires its
positive scale value and matching time/length units. An optional auxiliary
`direction` supplies a warm start. Without `objective_vector`, the target is the
negative baseline residual; otherwise that vector is the target in the frozen
objective coordinates. `normal` itself remains unregularized.

CG verifies its final true residual and reports iterations, normal actions,
convergence, residual norms and quadratic change. `cache_mb` bounds resident
incident checkpoints per rank; excess batches use temporary storage. A zero
budget forces spilling. The retained reduced-gradient source-batch workspace is
indivisible and sized by the batch; the deprecated `workspace_mb` is accepted and
ignored. `require_convergence` rejects an unconverged solve after emitting its
diagnostic artifacts.

For L2 fits, `gradient_checkpoints` optionally supplies an output prefix for a
fixed-tap background gradient at accepted CG iterates divisible by
`gradient_checkpoint_interval` (positive integer, default 1). Other iterations
skip checkpoint gradient computation entirely. This requires active
material controls and adds three propagation solves per source batch and
frequency at each checkpoint. Gradients use
`<prefix>_cg_<iteration>_<task>.h5`; matching JSON completion records contain the
iteration, gradient path, per-frequency unweighted data objective, frequency
weight, current inner residual norm (possibly recursive), normal-action count,
and checkpoint propagation-solve count. A checkpoint is **not** a stationary
reduced gradient. Sum its frequency covectors with the same band weights.
Checkpoint output does not change the CG stopping policy or Krylov recurrence.

For a reduced background gradient, select material blocks in `controls/active`
when creating the state, then use `solve`, `model_gradient=true`, and a top-level
`covector` output. This path requires a converged inner solve, waveform L2, Huber, or Student-t
objectives and the observed-data target (no `objective_vector`). Robust observed-data
solves update the true residual and IRLS metric, use backtracking, and check the
regularized objective gradient before publishing a reduced gradient. The
`max_outer_iterations`, `max_line_search`, `gradient_relative_tolerance`, and
`gradient_absolute_tolerance` solver settings control this iteration. The physical
covector includes both propagation dependencies, mixed scattering/material
partials, and supported source/receiver material derivatives. Acoustic and
isotropic, TI, and TTI elastic mixed volume coefficients are implemented. DPG keeps its test
metric fixed. Source/geometry controls and unsupported constitutive mixed
partials are rejected. For auxiliary-only actions, physical block selection does
not perturb the fixed background; `controls/active=[]` is sufficient.

For a reduced physical-model Gauss–Newton action, use `solve` with
`reduced_normal: {}`, an ordinary control-vector `direction`, and a top-level
`covector`. Select the same active material blocks when linearizing and applying
the action. This first solves for the optimal extension, then eliminates its
linearized response using the regularized extension normal. Time lags and spatial
offsets use the same operation. `reduced_normal` optionally overrides
`relative_tolerance`, `absolute_tolerance`, and `max_iterations` for the response
solve; omitted values inherit the extension solver settings.

With physical prediction derivative `G`, scaled extension derivative `B`, frozen
positive receiver curvature `W`, and extension-coordinate regularizer `D`, the
returned action is `G*WG - G*WB (B*WB + D)^(-1) B*WG`. For robust losses, `W` is the
positive IRLS approximation at the solved prediction. This is a Gauss–Newton
Schur approximation, not the exact reduced Hessian. It omits residual-weighted
second derivatives. Both the optimized extension and response solve must converge
before a physical covector is published, irrespective of `require_convergence`.
The report includes a `reduced_normal` object with method `gauss_newton_schur`,
response iterations, true residual, and convergence status. This option is
mutually exclusive with `model_gradient=true` and requires the observed-data
target. The same material-only and fixed DPG-metric restrictions apply. See the
[example](examples/fwi-operator-reduced-normal.json).

Full-dimensional waveform comparisons are required; relaxed assembly is
accepted as an approximation. Volume scattering excludes PML and
boundary coefficients. Factors and incident states are reused inside a request.
For one common fit, supply `extension/solver/frequency_weights` (one finite,
nonnegative weight per `f_list` entry, at least one positive) and launch with
`--frequency-groups N`, where `N` equals the complete frequency count. MPI ranks
must divide evenly into these contiguous groups. Each group retains one
frequency's factors and incident checkpoints. The native iteration sums weighted
data normals and right-hand sides, adds the regularizer once, and makes one
common convergence decision. Control basis, field order, physical units, axis,
scales and solver settings must agree; spatial partitions may differ. Routing
uses canonical global control IDs and checks one owner per frequency and ID.
Warm starts must represent the same physical tap vector on every group.

Task-suffixed solution files contain the same physical taps with per-task state
identities. Reports declare `scope: frequency_band` and `frequency_weights`;
`quadratic_objective` is the global band value. With `model_gradient`,
`data_objective` and the physical covector remain **per frequency**, unweighted.
Aggregate these with the supplied weights and add `regularization` from exactly
one report; never sum the per-task `reduced_objective` fields for the band.
The current shared path supports L2 inner fits and reduced gradients, not robust
losses or `reduced_normal`. The whole band must run together; partial reuse and
per-frequency retry are invalid. Without the weights/group launch, the native
contract retains independent frequency fits.
See the [solver guide](../../../src/Core/Manage/Simulation/extension.md).

The [iteration/composition design](../../internal/fs-extension-iteration-1/contract.md)
requires both Sauce-owned and FrequenSolve Python-owned iteration over the same
operator boundary, with frequency scheduling independent of driver choice.
Native callbacks and the MPI shared-band L2 launcher exist; a retained Python
session interface remains planned. This outline introduces no additional job fields. File-based external
quadratic iteration can use the current actions, but does not retain factors
across executable invocations or supply robust candidate reweighting.


## Joint background and coordinate reflectivity

`fwi_operator/reflectivity` selects a total-field, zero-lag Born prediction
`u + P S(m,r)u`. It supports the standard `linearize`, `jvp`, `vjp`, and `normal`
actions, with the same saved objective-state and joint control-vector contracts.
It is mutually exclusive with `extension`, whose auxiliary vectors and inner
solve have different semantics. The optimizer and frequency scheduling remain
FrequenSolve responsibilities.

Select `parameterization: vp_ip` for acoustic `(Vp,Ip)`, `vp_vs_ip` for elastic
`(Vp,Vs,Ip)`, or `ip_is_rho` for elastic `(Ip,Is,rho)`. Here `Ip=rho*Vp` and
`Is=rho*Vs` use real reference-axis velocities; anisotropy, orientation and
attenuation remain separate background quantities. Reflectivity values are
signed absolute coordinate perturbations in native solver units. They do not
inherit material transforms or bounds, and are not dimensionless interface
reflection coefficients.

Each field specifies `name`, one-based material `layer`, and one-based chart
`axis`. It registers `reflectivity.<name>` in the shared control registry before
active selection. Choose either an independent `control` object (the existing
hat or B-spline map with initial coefficients) or `basis`, the identifier of a
direct material control map. A borrowed basis starts at zero reflectivity;
`controls/state` may replace it. Borrowing the spatial map does not tie the
reflectivity values or coordinate meaning to that material property or its
active selection. A complete `controls/state_output` includes both background
and reflectivity. The registry fingerprint includes the coordinate map, layer,
spatial basis, baseline values and active selection.

The acquisition adapter samples total physical fields, includes supported
material receiver derivatives in both directions, and uses the existing waveform
objective linearization. Consequently `normal` is the data/objective GN/IRLS
normal, including background–reflectivity cross blocks. It is not the unweighted
physical-field normal of the lower-level native session. Robust objective
weights are frozen at the saved baseline, just as in ordinary FWI actions.

This workflow currently requires full-dimensional first-order acoustic or classic
elastic DPG, compiled Forms, native receiver channels, and the frozen
trial-to-test policy; relaxed assembly is accepted as an approximation. Coupled physics, Galerkin, 2.5D, axisymmetry, phase
objectives, explicit Dirichlet data, and sensitivity tapers are
rejected. Volume reflectivity excludes PML cells. The retained joint session holds
one indivisible source-batch workspace (the deprecated `workspace_mb` is accepted
and ignored); solver and acquisition buffers have their own owners.
One source batch is active at a time, and all propagations reuse the background
factors.

See [the example](examples/fwi-operator-reflectivity.json) and the
[native joint API](../../../docs/imaging/extended/joint_born.md).

## Retaining successful tasks for convergence retries

`preserve_task_outputs` is a boolean, default `false`. When true, packing retains
raw task files and a successful frequency task may be reused after changing
numerical solver settings. Failed tasks are still retried. Full original job and
simulation hashes remain in each producer result; the separate compatibility
fingerprint only controls this explicit reuse policy.

Changes to the physical simulation, acquisition, output request, task frequencies,
mesh controls, or structural solver fields (`grids`, `refinements`,
`refinement_flags`, `hp`, `galerkin_multigrid`, `relaxed_assembly`, `mode`)
invalidate reuse. The CLI PML frequency must also match. `--fresh` forces a new
solve under either policy. FrequenSolve refreshes external-input hashes when
saving/staging; save again after changing referenced input files.

The default packs completed shards, commits the packed dataset references to
their task results, then retires the duplicate shards and shared trace metadata.
Previously committed packed products survive unsuccessful replacement attempts
in both modes. An immutable segment still referenced by a current task or pack
is retained. This policy does not delete user-selected FWI checkpoint stems.

## Native model regularization callbacks

`control_sensitivities.Regularization` uses the `--smooth` entry point with an
explicit `input` full-model vector and `gradient` output. `operation="prepare"`
resolves weights/amplitude scales and writes `context`; `value` evaluates the
same native energy with that context; `gradient` returns the exact Tikhonov
coefficient covector and its energy (no mass inversion). The latter operation
rejects TV/TGV; on a zero-padded tangent it applies the Tikhonov Hessian.
`diagonal` returns the exact coefficient Hessian diagonal for Tikhonov, with
zero reported energy; its input values are otherwise unused. Both derivative operations
include constrained-basis assembly and use the frozen context weights.
`mass` applies the consistent material mass matrix to primal coefficients;
`mass_inverse` solves the mass system for an input coefficient covector.
`mass_diagonal` returns the positive diagonal of the consistent mass matrix,
including constrained-basis cross terms; its input values are otherwise unused.
These operations use native geometry and the constrained basis, ignore the
context's weights/amplitude, and report zero energy. The inverse uses `iterations`,
`relative_tolerance` and `absolute_tolerance`, and fails on nonconvergence.
Every operation other than `prepare`, including the mass operations, requires a
prepared `context`: it identifies the native basis, and a block whose recorded
identity differs from the bound control is rejected. Every operation reads a
full, finite `input` vector; non-finite values are rejected even where the
values are unused.
`proximal` minimizes metric fidelity plus
`tau` times that energy with full coefficient `lower`/`upper` bounds. Equal bounds
fix coefficients. `metric` is a positive coefficient diagonal. All vectors must
carry matching mesh `control_spaces` identities when applicable. They contain
full values, not zero-padded tangents. `result` follows
[fs-control-regularization-result-1](../../outputs/fs-control-regularization-result-1/contract.md).

TV/TGV use split-Bregman shrinkage for mesh, axis and tensor controls. `epsilon`
controls splitting, not epsilon smoothing of the norm. TGV uses first derivatives;
second-order scalar regularization requires a sufficiently high-degree spline.
The inversion driver adds this energy to its data objective once per model, after
frequency aggregation. Legacy `Smoothing` vector processing remains a distinct
operation with primal or dual input semantics.

For a coefficient field `u`, the native energies are
`alpha/2 * integral |D^p u|^2` (Tikhonov),
`sqrt(alpha) * integral |D^p u|` (TV), and
`min_w integral alpha1*|grad(u)-w| + alpha2*|sym(grad(w))|` (TGV).
Spatial axes use km and angular axes radians. Integrals use the native basis
and quadrature. Wavelength scaling sets `length=lambda*wavelength/(2*pi)`,
`alpha=length^(2*p)`, or `alpha1=length`, `alpha2=tgv_ratio*length^2`.
Explicit weights override these defaults.

The prepared context records each included material block's basis identity,
resolved weights and amplitude `a`. Normalization defaults on for wavelength
weights and off for explicit weights; when enabled, `a` is the maximum absolute
prepared input coefficient (one for a zero block). The energy is then
`a^2 * R(u/a)`. The same context and regularizer configuration must be reused
throughout a stage, including after checkpoint restoration. The SDK supplies
`u=m-reference` for a reference-state regularizer; the callback does not subtract
a reference itself.

The proximal objective is `0.5*(u-input)^T metric*(u-input) + tau*R(u)` with
coefficient bounds. The reported value excludes both fidelity and `tau`.
`input_role` is ignored here: all inputs are primal full coefficients, including
fixed values. `prepare` and `value` return the supplied input in `gradient`;
`proximal` returns the constrained model. Relative output paths resolve under
the result directory; input vectors and input contexts use the control-vector
input lookup rules above. `Regularization` takes precedence if `Smoothing` is
also present; frequency aggregation and scalar data-objective aggregation are
not performed by this callback.

FrequenSolve's `Tikhonov`, `TV` and `TGV` objects configure these callbacks;
they do not implement separate Python discretizations or derivatives.
FWI preserves its requested smooth optimizer for Tikhonov using `gradient`.
TV/TGV and native LSRTM terms use composite proximal-gradient backtracking.
Explicit custom smooth terms remain SDK-owned and may be added to that term.

### Shared receiver groups

Receiver actions cover all configured imaging receiver groups in one PDE
hierarchy. A single group retains `fs-receiver-state-1` /
`fs-receiver-vector-1`; multiple groups use the hash-bound
`fs-receiver-state-bundle-1` / `fs-receiver-vector-bundle-1` collection manifests.
Each member retains its own physical keys, units and observation identity.
Members share the model, acquisition, frequency, partition and field checkpoints.
`receiver_jvp` exports all group tangents after shared solves. `receiver_vjp`
loads all group duals and sums their receiver loads before each common adjoint
solve. The control direction/covector is bound to the collection fingerprint.

## Control-mesh adaptation callback

`control_sensitivities.MeshAdaptation` is an explicit single-rank `--smooth`
setup request. `input` supplies the full accepted material coefficients with their
old basis identities; `gradient` names the transferred native coefficient file.
The request contains `source_identity`, positive `frequency` (Hz), positive
`averaging_wavelengths` (window half-width), `model.property_spaces` declarations
for the spaces to replace, and the JSON `result` path. Old and new spaces use the
same initial geometry, material groups and property transforms. Unlisted spaces
remain unchanged. Output basis identities refer to the new property artifacts.

The sizing-only reference averages recovered slowness without crossing material
interfaces; its physical window is fixed before refinement from the old material's
volume-weighted harmonic-mean wavelength. Five-point Gauss quadrature per axis
approximates a truncated Gaussian window. `transfer="nodal"` (default) interpolates
coefficient updates. `transfer="l2"` integrates the source and target constrained
bases on their cell intersections and solves the target projection. Optional
`smoothing_length_m >= 0` adds that physical length squared times the target
stiffness operator; nonzero smoothing requires `transfer="l2"`. Zero length is
ordinary L2 projection. Both refinement and coarsening are supported, with native
geometry and a checked matrix-free solve. Constants are preserved; general
coarsening loses unresolved structure. Alternatively, `smoothing_wavelengths`
(nonnegative, L2 only, with zero fixed length) sets the local smoothing length to
`fraction * accepted_wavespeed(x) / smoothing_frequency_hz`. The frequency defaults
to the requested sizing frequency and must be positive. The source model is
restored before sampling the wavespeed; the resulting coefficient is frozen during
the linear solve and integrated inside the stiffness operator. For acoustics the
speed is Vp; elastic materials use the native minimum propagating wavespeed.
Positive smoothing acts on model
coefficients (log updates for log controls), retaining the reference. This is not
a gradient transfer or an optimization regularization callback. Artifacts must use immutable paths keyed
by the source state and sizing policy; retain them for all stage evaluations and
restarts. The result follows `fs-control-mesh-adaptation-result-1`.

The default `control_sensitivities.quadrature: "auto"` evaluates tensor-hat
**volume sensitivities at tensor nodes without cell-volume weights** for RTM and
FWI pullbacks (`linearize`, `vjp`, `receiver_vjp`, and gradient-only `wri`). This
avoids missing fine control nodes. The gridded-image mapper supports curved
wavefield elements; cached reference points and shared-element averaging prevent
double counting. These are approximate nodal sensitivity values, not integrated
coefficient covectors. Densities use physical km coordinates (per km^D),
independent of solver nondimensionalization. Their Euclidean dot product is not an exact directional
derivative of the discrete objective.

Explicit `"tensor_points"` requests the same sampling. Explicit `"wavefield"`
retains the discrete coefficient gradient for derivative/transpose tests. Auto
keeps native quadrature for depth-only and other non-tensor layers, Born/JVP,
normal actions, WRI curvature/diagonal actions, `fwi_operator.extension`,
intersected assembly, and jobs with active geometry controls. Forward assembly
and face terms remain unchanged. Sampling requires full-dimensional,
axis-aligned Cartesian tensor controls sharing one layout per material layer.
Auto integrates any other layer natively, including layers that mix tensor and
other active controls and every layer sharing a tensor parameter with such a
layer. Explicit `"tensor_points"` rejects those layers, active geometry
controls, and `fwi_operator.extension`. Depth-strip integration is deferred.

Diagnostics include `tensor_nodal_sensitivity`, `control_quadrature_points`,
`control_quadrature_original_points`, `control_quadrature_cache_bytes`,
`control_quadrature_setup_us`, and `control_quadrature_reuse`.
`tensor_nodal_sensitivity` is `1` only when at least one layer was sampled.
Point and cache-byte counts are summed over ranks; `control_quadrature_reuse` is
the maximum per-rank reuse count. Cache bytes count reference-point and
averaging-weight payload only. Smoothing of generated tensor sensitivities uses
nodal (primal) input in sampled layers only; explicit input vectors retain their
requested input role.

## Internal root preparation

The experimental top-level `patches` field is SDK-only metadata for saved
frequency-domain forward wrappers. It records explicit roots or automatic shot
grouping, scalar radial or per-axis box aperture, buffering, depth and PML policy.
Run these wrappers through `site.run`; direct site submission is rejected.
The SDK prepares ordinary native child jobs without this field, preserving the
source catalog and selecting `Acquisition/active_sources` plus sparse receiver
rows. Receiver groups with no retained rows are omitted from that child.

`patch_prepare` runs one geometry-only operation with a `PatchPreparation`
request (`fs-patch-preparation-1`) and a positive stage `f_list`. It publishes
the parent snapshot and `fs-patch-geometry-1` report without wave solves.

Experimental `fwi_operator/controls/stage_mesh` captures or replays a verified
[frequency mesh companion](../../internal/fs-stage-mesh-1/contract.md). It requires
`pml_stage` and an explicit candidate `state`. Capture uses the canonical stage
baseline with `action: linearize` and stops before wave solves. Replay preserves
the captured h/p mesh and solver hierarchy; changed execution context is an error.
